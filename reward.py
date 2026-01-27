import re
from tqdm import tqdm
from consts import REASONING_END, REASONING_START, SOLUTION_START, SOLUTION_END


def formatting_reward_func(completions, **kwargs):
    thinking_pattern = f"{REASONING_START}(.*?){REASONING_END}"
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"
    scores = []
    for completion in tqdm(completions, desc="Computing formatting reward"):
        score = 0
        thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
        answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
        if len(thinking_matches) == 1:
            score += 1.0
        if len(answer_matches) == 1:
            score += 1.0
        scores.append(score)
    return scores


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    answer_pattern = f"{SOLUTION_START}(.*?){SOLUTION_END}"

    responses = [
        re.findall(answer_pattern, completion, re.DOTALL)
        for completion in tqdm(completions, desc="Extracting responses for correctness")
    ]
    q = prompts[0]

    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:{completions[0]}",
    )
    return [
        2.0 if len(r) == 1 and a == r[0].replace("\n", "") else 0.0
        for r, a in tqdm(
            zip(responses, answer), desc="Checking correctness", total=len(responses)
        )
    ]
