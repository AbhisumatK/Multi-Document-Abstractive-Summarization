import math
import os
import re
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer


project_root = os.getcwd()
sys.path.append(os.path.join(project_root, "Agent_1_Packing_Agent"))
sys.path.append(os.path.join(project_root, "Agent_2_Document_Agregation_Agent"))
sys.path.append(os.path.join(project_root, "Agent_3_Faithful_Generator_Agent"))

from Agent_1_Packing_Agent.src.model.summarizer import BertSum
from Agent_1_Packing_Agent.src.training.rl_policy import actor_critic_loss, sample_sentence_actions
from Agent_2_Document_Agregation_Agent.src.model.aggregation_agent import CrossDocumentAggregationAgent
from Agent_2_Document_Agregation_Agent.src.utils.entity_utils import extract_entities, get_entity_alignment_matrix
from Agent_3_Faithful_Generator_Agent.src.model.generator_agent import FaithfulGeneratorAgent
from Agent_3_Faithful_Generator_Agent.src.utils.reward_utils import SummarizationReward


def split_sentences(documents):
    abbreviations = {
        "Inc.": "Inc<PERIOD>",
        "Ltd.": "Ltd<PERIOD>",
        "Corp.": "Corp<PERIOD>",
        "Co.": "Co<PERIOD>",
        "Dr.": "Dr<PERIOD>",
        "Mr.": "Mr<PERIOD>",
        "Mrs.": "Mrs<PERIOD>",
        "Ms.": "Ms<PERIOD>",
        "U.S.": "U<PERIOD>S<PERIOD>",
        "U.K.": "U<PERIOD>K<PERIOD>",
    }

    sentences = []
    for document in documents:
        protected_document = document
        for abbreviation, replacement in abbreviations.items():
            protected_document = protected_document.replace(abbreviation, replacement)

        sentences.extend([
            sentence.replace("<PERIOD>", ".").strip()
            for sentence in re.split(r"(?<=[.!?])\s+", protected_document)
            if sentence.strip()
        ])
    return sentences


class MARLMdsTrainer:
    """
    Minimal shared-reward training scaffold for the MARL-MDS work plan.

    The current demo generator is extractive for faithfulness, so this trainer
    updates the packing policy with the final summary reward. A trained
    abstractive checkpoint can later plug into Agent 3's differentiable
    teacher-forcing path.
    """
    def __init__(
        self,
        device=None,
        model_name="bert-base-uncased",
        learning_rate=2e-5,
        local_files_only=True,
        checkpoint_path=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)

        self.agent1 = BertSum(model_name=model_name, local_files_only=local_files_only).to(self.device)
        self.agent2 = CrossDocumentAggregationAgent(d_model=768).to(self.device)
        self.agent3 = FaithfulGeneratorAgent(local_files_only=local_files_only, bert_dim=768).to(self.device)
        self.reward_fn = SummarizationReward()

        self.optimizer = optim.Adam(
            list(self.agent1.parameters()) + list(self.agent2.parameters()) + list(self.agent3.parameters()),
            lr=learning_rate,
        )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def encode_sentences(self, sentences):
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.agent1.bert(**inputs)
            sentence_embeddings = outputs.last_hidden_state[:, 0, :]

        return sentence_embeddings

    def run_episode(self, documents, reference_summary=None, compression_ratio=0.5, max_length=150, summary_mode="abstractive", target_sentences=None, source_documents=None):
        sentences = split_sentences(documents)
        if not sentences:
            raise ValueError("No sentences found in input documents.")

        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        joined_text = " ".join([f"{cls_token} {sentence} {sep_token}" for sentence in sentences])
        encoded = self.tokenizer(
            joined_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        cls_positions = (encoded["input_ids"][0] == self.tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
        if cls_positions.numel() == 0:
            raise ValueError("No CLS positions found after tokenization.")

        sentences = sentences[:cls_positions.numel()]
        cls_positions = cls_positions.unsqueeze(0)

        salience_scores, state_values = self.agent1.actor_critic(
            encoded["input_ids"],
            encoded["attention_mask"],
            cls_positions,
        )
        salience_scores = salience_scores[:, :len(sentences)]

        k = max(1, math.ceil(len(sentences) * compression_ratio))  # Remove cap to allow more sentences
        selected_indices, log_probs, entropy = sample_sentence_actions(salience_scores, k=k)
        selected_indices_list = selected_indices[0].detach().cpu().tolist()
        selected_indices_list.sort()
        selected_sentences = [sentences[index] for index in selected_indices_list]

        sentence_embeddings = self.encode_sentences(selected_sentences).unsqueeze(0)
        entity_bias = get_entity_alignment_matrix(extract_entities(selected_sentences), device=self.device).unsqueeze(0)
        fused_context = self.agent2(sentence_embeddings, entity_bias=entity_bias)

        # Use the selected summary mode
        mode = summary_mode
        try:
            summary, policy_logits_a3, value_a3 = self.agent3.generate_faithful(
                fused_context,
                source_sentences=selected_sentences,
                reference=reference_summary,
                max_length=max_length,
                mode=mode,
                use_rl=(reference_summary is not None),  # Only use RL if reference is provided
                target_sentences=target_sentences,
                source_documents=documents if source_documents is None else source_documents
            )
            summary = summary[0]
        except Exception as e:
            print(f"{mode} generation failed: {e}, falling back to extractive")
            summary, _, _ = self.agent3.generate_faithful(
                fused_context,
                source_sentences=selected_sentences,
                reference=reference_summary,
                max_length=max_length,
                mode="extractive",
                use_rl=False,
                target_sentences=target_sentences,
                source_documents=documents if source_documents is None else source_documents
            )
            summary = summary[0]
            policy_logits_a3 = None
            value_a3 = None
        
        # Only compute reward and RL loss if reference is provided
        if reference_summary is not None:
            reward_value = self.reward_fn.compute_reward(summary, reference_summary, source_text=" ".join(documents))
            rewards = torch.tensor([reward_value], device=self.device)

            # Compute RL loss for Agent 1
            rl_loss_a1, metrics_a1 = actor_critic_loss(log_probs, state_values, rewards, entropy)
            
            # Compute RL loss for Agent 3 (generation parameter control)
            rl_loss_a3 = torch.tensor(0.0, device=self.device)
            if policy_logits_a3 is not None and value_a3 is not None:
                # Sample action from policy logits
                action_probs = torch.softmax(policy_logits_a3, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob_a3 = action_dist.log_prob(action)
                
                # Compute advantage (reward - value)
                advantage = rewards - value_a3.detach()
                
                # Policy loss for Agent 3
                rl_loss_a3 = -(log_prob_a3 * advantage).mean()
                
                # Value loss for Agent 3
                value_loss = nn.functional.mse_loss(value_a3, rewards)
                rl_loss_a3 = rl_loss_a3 + 0.5 * value_loss

            # Tokenize reference_summary for Agent 3 teacher forced loss
            target_tokens = self.agent3.tokenizer(
                reference_summary,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(self.device)
            target_ids = target_tokens["input_ids"]

            loss_a3_supervised, logits_a3 = self.agent3(fused_context, target_ids=target_ids)

            # Combine losses
            total_loss = rl_loss_a1 + rl_loss_a3 + loss_a3_supervised

            metrics = metrics_a1.copy()
            metrics["loss_a3_supervised"] = loss_a3_supervised.item()
            metrics["rl_loss_a1"] = rl_loss_a1.item()
            metrics["rl_loss_a3"] = rl_loss_a3.item()
            metrics["reward"] = reward_value
        else:
            # Inference mode without reference - no RL loss
            total_loss = torch.tensor(0.0, device=self.device)
            metrics = {
                "reward": 0.0
            }

        return total_loss, {
            **metrics,
            "summary": summary,
            "selected_sentences": selected_sentences,
        }

    def train_step(self, documents, reference_summary):
        self.agent1.train()
        self.agent2.train()
        self.agent3.train()

        self.optimizer.zero_grad()
        loss, metrics = self.run_episode(documents, reference_summary)
        loss.backward()
        self.optimizer.step()

        metrics["loss"] = loss.item()
        return metrics

    def save_checkpoint(self, checkpoint_path):
        torch.save(
            {
                "agent1": self.agent1.state_dict(),
                "agent2": self.agent2.state_dict(),
                "agent3": self.agent3.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.agent1.load_state_dict(checkpoint["agent1"])
        self.agent2.load_state_dict(checkpoint["agent2"])
        self.agent3.load_state_dict(checkpoint["agent3"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Checkpoint loaded from {checkpoint_path}")


if __name__ == "__main__":
    docs = [
        "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It is the world's largest technology company by revenue.",
        "Steve Jobs and Steve Wozniak founded Apple in 1976. The company is famous for the iPhone and Mac computers.",
        "Recent reports suggest Apple is investing heavily in artificial intelligence and autonomous vehicles to expand its product line.",
    ]

    # Use trained checkpoint by default
    checkpoint_path = os.path.join(project_root, "checkpoints", "marl_mds_multinews.pt")
    
    trainer = MARLMdsTrainer(checkpoint_path=checkpoint_path)
    
    # Run inference without reference summary
    loss, metrics = trainer.run_episode(docs)
    print("--- MARL Inference with Trained Model ---")
    print("Summary:", metrics["summary"])
    print(f"Selected {len(metrics['selected_sentences'])} sentences")
