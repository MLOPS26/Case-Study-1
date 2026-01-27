import gradio as gr
import jax
import jax.numpy as jnp
from jax import random
from jax.random import PRNGKey
import json
from globals import Char, State, UserInfo


from thompson import (
    init_thompson,
    recommend_characters,
    update_posterior,
    compute_reward,
    construct_feats,
)
from transformers import AutoTokenizer, AutoModelForCausalLM


class LMCharacterKnowledge:
    def __init__(self, model_name: str, game_name: str):
        self.game_name = game_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.prompt = [
            {
                "role": "system",
                "content": "You are a knowledgeable bastion of fighting game knowledge. Your goal is to answer questions as best as possible about the game you are asked about.",
            }
        ]
        self.cache = {}

    def ask_lm(self, prompt, max_tok: int = 4096):
        try:
            messages = self.prompt + [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            result = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            print(result)
            return result
        except Exception as e:
            print(f"Couldn't query{self.model}, error: {e}")

    def get_roster(self) -> list[str]:
        cache_key = f"roster_{self.game_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        roster_prompt = f"""
        List ALL playable characters in {self.game_name}. Return a structured json array of character names, nothing else at all. 
        Example format is : ["Ryu", "Ken", "Chun Li", "Akuma"]
        """

        response = self.ask_lm(roster_prompt)

        try:
            start = response.find("[")
            end = response.find("]") + 1

            if start != -1 and end > start:
                roster = json.loads(response[start:end])
                self.cache[cache_key] = roster
                return roster
        except:
            # TODO: handle errors here way better
            pass

        return ["Ryu", "Ken", "Luke"]

    def get_character_data(self, char_name: str) -> dict:
        cache_key = f"char_{self.game_name}_{char_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        char_data_prompt = f"""
        for the character {char_name} in the game {
            self.game_name
        }, 
        provide some statistics in explicit JSON format:
           
        Example format:
        {{
            "difficulty": 0.7,
            "execution_barrier": 0.6,
            "neutral_intensity": 0.5,
            "tier": 0.8,
            "archetypes": {{
                "rushdown": 0.8,
                "zoner": 0.1,
                "grappler": 0.0,
                "all_rounder": 0.1,
                "setplay": 0.0,
                "footsies": 0.0
            }}
        }}

        Replace ALL values with actual numbers for {char_name}. Return ONLY the JSON object, nothing else.
        """

        response = self.ask_lm(char_data_prompt, max_tok=300)
        print(f"Raw response for {char_name}: {response}")

        try:
            start = response.find("{")
            if start == -1:
                raise ValueError("No opening brace found")
            
            brace_count = 0
            end = -1
            for i in range(start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end == -1:
                raise ValueError("No matching closing brace found")
            
            json_str = response[start:end]
            print(f"Extracted JSON: {json_str}")
            
            data = json.loads(json_str)
            
            required_keys = ["difficulty", "execution_barrier", "neutral_intensity", "tier", "archetypes"]
            if not all(key in data for key in required_keys):
                raise ValueError(f"Missing required keys in parsed data")
            
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            print(f"Couldn't parse {char_name}'s data: {e}")
            print(f"Response was: {response[:200]}...") 

        return {
            "difficulty": 0.5,
            "execution_barrier": 0.5,
            "neutral_intensity": 0.5,
            "tier": 0.5,
            "archetypes": {
                "rushdown": 0.3,
                "zoner": 0.3,
                "grappler": 0.1,
                "all_rounder": 0.2,
                "setplay": 0.05,
                "footsies": 0.05,
            },
        }

    def build_roster(self) -> tuple[list[Char], list[str]]:
        roster = self.get_roster()
        chars = []

        for i, char_name in enumerate(roster):
            data = self.get_character_data(char_name)
            archetype_order = [
                "rushdown",
                "zoner",
                "grappler",
                "all_rounder",
                "setplay",
                "footsies",
            ]
            archetype_vec = jnp.array(
                [data["archetypes"].get(a, 0.0) for a in archetype_order]
            )

            archetype_vec = archetype_vec / (jnp.sum(archetype_vec) + 1e-8)

            char = Char(
                difficulty=data["difficulty"],
                archetype_vec=archetype_vec,
                execution_level=data["execution_barrier"],
                neutral_required=data["neutral_intensity"],
                tier=data["tier"],
            )
            chars.append(char)

        batched_chars = Char(
            difficulty=jnp.array([c.difficulty for c in chars]),
            archetype_vec=jnp.stack([c.archetype_vec for c in chars]),
            execution_level=jnp.array([c.execution_level for c in chars]),
            neutral_required=jnp.array([c.neutral_required for c in chars]),
            tier=jnp.array([c.tier for c in chars]),
        )

        return batched_chars, roster


class FGRecommender:
    def __init__(self):
        self.lm = None
        self.chars = None
        self.roster = None
        self.state = None
        self.user = None
        self.key = PRNGKey(67)
        self.n_archetypes = 6
        self.history = []

    def init_game(self, game_name: str) -> str:
        if not game_name.strip():
            return "please enter name of game"

        try:
            self.lm = LMCharacterKnowledge(model_name="LiquidAI/LFM2-350M", game_name = game_name)
            self.chars, self.roster = self.lm.build_roster()

            n_chars = len(self.roster)
            feature_dim = 17

            self.state = init_thompson(n_chars, feature_dim)

            self.user = UserInfo(
                skill_level=0.3,
                games_played=0,
                chars_attempted_mask=jnp.zeros(n_chars),
                wr=jnp.ones(n_chars) * 0.5,
                playtime=jnp.zeros(n_chars),
                pref_archetype=jnp.zeros(self.n_archetypes),
            )

            return f"loaded {n_chars} from {game_name}"
        except Exception as e:
            return f"Error: {e}"

    def get_recs(self, top_k: int = 5) -> tuple[str, str] | str:
        if self.state is None:
            return "please init game"

        self.key, subkey = random.split(self.key)

        sel, sample_rewards = recommend_characters(
            subkey,
            self.state,
            self.user,
            self.chars,
            len(self.roster),
            top_k=top_k,
            diversity_threshold=0.75,
        )

        recommend_text = "## Recommended Chars: \n\n"
        for i, char_idx in enumerate(sel):
            char_idx = int(char_idx)
            if char_idx < 0:
                continue

            char_name = self.roster[char_idx]
            reward = float(sample_rewards[char_idx])
            tried = bool(self.user.chars_attempted_mask[char_idx] > 0.5)

            status = "NEW" if not tried else "TRIED"

            recommend_text += f"### {i + 1}. {char_name} {status} \n"
            recommend_text += f"expected_reward: {reward: .4f} \n"
            recommend_text += f"difficulty: {self.chars.difficulty[char_idx]:.2f}\n"
            recommend_text += f" Tier: {self.chars.tier[char_idx]:.2f}\n\n"

        char_opts = [self.roster[int(idx)] for idx in sel if idx >= 0]

        return recommend_text, gr.Dropdown(
            choices=char_opts, value=char_opts[0] if char_opts else None
        )

    def record_feedback(
        self, char_name: str, won: bool, rating: float, playtime: float
    ) -> str:
        if self.state is None or char_name is None:
            return "get recs first"

        try:
            char_idx = self.roster.index(char_name)
        except ValueError:
            return f"char {char_name} not found"

        sel_char_obj = jax.tree.map(lambda x: x[char_idx], self.chars)
        feats = construct_feats(self.user, sel_char_obj, char_idx)

        reward = compute_reward(
            won=won, completed=True, rating=rating, playtime_mins=playtime
        )
        self.user = self.user._replace(
            games_played=self.user.games_played + 1,
            chars_attempted_mask=self.user.chars_attempted_mask.at[char_idx].set(1),
            wr=self.user.wr.at[char_idx].set(
                0.8 * self.user.wr[char_idx] + 0.2 * float(won)
            ),
            playtime=self.user.playtime.at[char_idx].add(playtime),
        )

        self.history.append(
            {
                "character": char_name,
                "won": won,
                "rating": rating,
                "reward": float(reward),
            }
        )

        return f"recorded {char_name}'s feedback! Reward was {reward:.4f}"

    def get_stats(self) -> str:
        if self.user is None:
            return "no stats lol. play some games u scrub"

        tried = int(jnp.sum(self.user.chars_attempted_mask))
        total = len(self.roster)
        avg_wr = float(jnp.mean(self.user.wr))

        stats = f"""## Your Stats

        - **Games played:** {self.user.games_played}
        - **Characters tried:** {tried}/{total}
        - **Average win rate:** {avg_wr:.1%}
        - **Skill level:** {self.user.skill_level:.2f}
        """
        if tried > 0:
            top_indices = jnp.argsort(-self.user.playtime)[:5]
            stats += "\n###Most Played:\n"
            for idx in top_indices:
                idx = int(idx)
                playtime = float(self.user.playtime[idx])
                if playtime > 0:
                    char_name = self.roster[idx]
                    wr = float(self.user.wr[idx])
                    stats += f"- **{char_name}**: {playtime:.0f}m, {wr:.1%} WR\n"

        return stats

#
app = FGRecommender()


def create_ui():
    with gr.Blocks(
        title="Fighting Game Character Recommender", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("#  Fighting Game Character Recommender")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Setup")
                game_input = gr.Textbox(
                    label="Game Name",
                    placeholder="e.g., Street Fighter 6, Guilty Gear Strive",
                    value="Street Fighter 6",
                )
                init_btn = gr.Button("Initialize Game", variant="primary")
                init_output = gr.Markdown()

                gr.Markdown("### User Profile")
                skill_slider = gr.Slider(0.0, 1.0, value=0.3, label="Skill Level")

                stats_display = gr.Markdown("No stats yet")
                refresh_stats_btn = gr.Button("Refresh Stats")

            with gr.Column(scale=2):
                gr.Markdown("### Recommendations")
                top_k_slider = gr.Slider(
                    1, 5, value=3, step=1, label="Number of Recommendations"
                )
                get_rec_btn = gr.Button("Get Recommendations", variant="primary")
                rec_output = gr.Markdown()

                gr.Markdown("### Record Feedback")
                with gr.Row():
                    char_dropdown = gr.Dropdown(label="Character Played", choices=[])
                    won_checkbox = gr.Checkbox(label="Won?", value=False)

                with gr.Row():
                    rating_slider = gr.Slider(
                        1, 5, value=3, step=0.5, label="Rating (1-5)"
                    )
                    playtime_slider = gr.Slider(
                        5, 60, value=20, step=5, label="Playtime (minutes)"
                    )

                submit_btn = gr.Button("Submit Feedback", variant="secondary")
                feedback_output = gr.Markdown()

        def init_game(game_name):
            result = app.init_game(game_name)
            stats = app.get_stats()
            return result, stats

        init_btn.click(
            init_game, inputs=[game_input], outputs=[init_output, stats_display]
        )

        get_rec_btn.click(
            lambda k: app.get_recs(int(k)),
            inputs=[top_k_slider],
            outputs=[rec_output, char_dropdown],
        )

        submit_btn.click(
            app.record_feedback,
            inputs=[char_dropdown, won_checkbox, rating_slider, playtime_slider],
            outputs=[feedback_output],
        )

        refresh_stats_btn.click(app.get_stats, outputs=[stats_display])

    return demo

#
if __name__ == "__main__":

    demo = create_ui()
    demo.launch()

