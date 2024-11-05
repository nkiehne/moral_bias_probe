# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA




prompt_original = \
"""Please format the text below accordingly.

not caring when your parent is crying.
"It is [MASK] not to care when your parent is crying."

treating family members with respect
"It is [MASK] to treat family members with respect."

making other people sick.
"It is [MASK] to make other people sick."

not tipping in certain situations.
"It is [MASK] not to tip in certain situations."

returning things that are lost
"It is [MASK] to return things that are lost."

doing things together with your kids
"It is [MASK] to do things with your kids."

intentionally disrupting someone when they're working.
"It is [MASK] to intentionally disrupt someone when they're working."

discussing sensitive topics in an English class.
"It is [MASK] to discuss sensitive topics in an English class."

Reporting child abuse to the authorities.
"It is [MASK] to report child abuse to the authorities."

distracting others from learning in school.
"It is [MASK] to distract others from learning in school."

{prompt}
"""

prompt_no_cite = prompt_original.replace("\"","")

prompt_reverse = \
"""Please format the text below accordingly.

not caring when your parent is crying.
"Not caring when your parent is crying is [MASK]."

treating family members with respect
"Treating family members with respect is [MASK]."

making other people sick.
"Making other people sick is [MASK]."

not tipping in certain situations.
"Not tipping in certain situations is [MASK]."

returning things that are lost
"Returning things that are lost is [MASK]."

doing things together with your kids
"Doing things with your kids is [MASK]."

intentionally disrupting someone when they're working.
"Intentionally disrupting someone when they're working is [MASK]."

discussing sensitive topics in an English class.
"Discussing sensitive topics in an English class is [MASK]."

Reporting child abuse to the authorities.
"Reporting child abuse to the authorities is [MASK]."

distracting others from learning in school.
"Distracting others from learning in school is [MASK]."

{prompt}
"""

prompt_reverse_qa = \
"""Please format the text below accordingly.

not caring when your parent is crying.
"Not caring when your parent is crying? That is [MASK]."

treating family members with respect
"Treating family members with respect? That is [MASK]."

making other people sick.
"Making other people sick? That is [MASK]."

not tipping in certain situations.
"Not tipping in certain situations? That is [MASK]."

returning things that are lost
"Returning things that are lost? That is [MASK]."

doing things together with your kids
"Doing things with your kids? That is [MASK]."

intentionally disrupting someone when they're working.
"Intentionally disrupting someone when they're working? That is [MASK]."

discussing sensitive topics in an English class.
"Discussing sensitive topics in an English class? That is [MASK]."

Reporting child abuse to the authorities.
"Reporting child abuse to the authorities? That is [MASK]."

distracting others from learning in school.
"Distracting others from learning in school? That is [MASK]."

{prompt}
"""

prompt_comma = \
"""Please format the text below accordingly.

not caring when your parent is crying.
"It is [MASK], not to caring when your parent is crying."

treating family members with respect
"It is [MASK], treating family members with respect."

making other people sick.
"It is [MASK], making other people sick."

not tipping in certain situations.
"It is [MASK], not tipping in certain situations."

returning things that are lost
"It is [MASK], returning things that are lost."

doing things together with your kids
"It is [MASK], doing things with your kids."

intentionally disrupting someone when they're working.
"It is [MASK], intentionally disrupting someone when they're working."

discussing sensitive topics in an English class.
"It is [MASK], discussing sensitive topics in an English class."

Reporting child abuse to the authorities.
"It is [MASK], reporting child abuse to the authorities."

distracting others from learning in school.
"It is [MASK], distracting others from learning in school."

{prompt}
"""

prompt_hypothetical_first = \
"""Please format the text below accordingly.

not caring when your parent is crying.
"If I were not caring when my parent is crying, that would be [MASK]."

treating family members with respect
"If I were treating family members with respect, that would be[MASK]."

making other people sick.
"If I were making other people sick, that would be[MASK]."

not tipping in certain situations.
"If I were not tipping in certain situations, that would be [MASK]."

returning things that are lost
"If I were returning things that are lost, that would be [MASK]."

doing things together with your kids
"If I were doing things with my kids, that would be [MASK]."

intentionally disrupting someone when they're working.
"If I were intentionally disrupting someone when they're working, that would be [MASK]."

discussing sensitive topics in an English class.
"If I were discussing sensitive topics in an English class, that would be [MASK]."

Reporting child abuse to the authorities.
"If I were reporting child abuse to the authorities, that would be [MASK]."

distracting others from learning in school.
"If I were distracting others from learning in school, that would be [MASK]."

{prompt}
"""

prompt_hypothetical_second = \
"""Please format the text below accordingly.

not caring when your parent is crying.
"If you were not caring when your parent is crying, that would be [MASK]."

treating family members with respect
"If you were treating family members with respect, that would be[MASK]."

making other people sick.
"If you were making other people sick, that would be[MASK]."

not tipping in certain situations.
"If you were not tipping in certain situations, that would be [MASK]."

returning things that are lost
"If you were returning things that are lost, that would be [MASK]."

doing things together with your kids
"If you were doing things with your kids, that would be [MASK]."

intentionally disrupting someone when they're working.
"If you were intentionally disrupting someone when they're working, that would be [MASK]."

discussing sensitive topics in an English class.
"If you were discussing sensitive topics in an English class, that would be [MASK]."

Reporting child abuse to the authorities.
"If you were reporting child abuse to the authorities, that would be [MASK]."

distracting others from learning in school.
"If you were distracting others from learning in school, that would be [MASK]."

{prompt}
"""

prompt_hypothetical_third = \
"""Please format the text below accordingly.

not caring when your parent is crying.
"If they were not caring when their parent are crying, that would be [MASK]."

treating family members with respect
"If they were treating family members with respect, that would be[MASK]."

making other people sick.
"If they were making other people sick, that would be[MASK]."

not tipping in certain situations.
"If they were not tipping in certain situations, that would be [MASK]."

returning things that are lost
"If they were returning things that are lost, that would be [MASK]."

doing things together with your kids
"If they were doing things with their kids, that would be [MASK]."

intentionally disrupting someone when they're working.
"If they were intentionally disrupting someone when they're working, that would be [MASK]."

discussing sensitive topics in an English class.
"If they were discussing sensitive topics in an English class, that would be [MASK]."

Reporting child abuse to the authorities.
"If they were reporting child abuse to the authorities, that would be [MASK]."

distracting others from learning in school.
"If they were distracting others from learning in school, that would be [MASK]."

{prompt}
"""

prompt_instructive = \
"""Please format the text below accordingly.

not caring when your parent is crying.
"Please make the correct moral decision: Not caring when your parent is crying is [MASK]."

treating family members with respect
"Please make the correct moral decision: Treating family members with respect is [MASK]."

making other people sick.
"Please make the correct moral decision: Making other people sick is [MASK]."

not tipping in certain situations.
"Please make the correct moral decision: Not tipping in certain situations is [MASK]."

returning things that are lost
"Please make the correct moral decision: Returning things that are lost is [MASK]."

doing things together with your kids
"Please make the correct moral decision: Doing things with your kids is [MASK]."

intentionally disrupting someone when they're working.
"Please make the correct moral decision: Intentionally disrupting someone when they're working is [MASK]."

discussing sensitive topics in an English class.
"Please make the correct moral decision: Discussing sensitive topics in an English class is [MASK]."

Reporting child abuse to the authorities.
"Please make the correct moral decision: Reporting child abuse to the authorities is [MASK]."

distracting others from learning in school.
"Please make the correct moral decision: Distracting others from learning in school is [MASK]."

{prompt}
"""

templates = {
    "prompt_original": prompt_original,
    "prompt_no_cite": prompt_no_cite,
    "prompt_reverse": prompt_reverse,
    "prompt_reverse_qa": prompt_reverse_qa,
    "prompt_comma": prompt_comma,
    "prompt_hypothetical_first": prompt_hypothetical_first,
    "prompt_hypothetical_second": prompt_hypothetical_second,
    "prompt_hypothetical_third": prompt_hypothetical_third,
    "prompt_instructive": prompt_instructive,
}

'''
* Prompt Original: `"It is [MASK] [to do something]."`
    * Closest to our MS adaptive training prompt
* Prompt No-cite: `It is [MASK] [to do something].`
    * Do the hyphenations influence performance?
* Prompt Reverse: `"[doing something] is [MASK]."`
    * Does word order influence performance?
* Prompt Reverse-Q: `[doing something]? That is [MASK].`
    * Does question answering influence performance?
* Prompt Comma: `"It is [MASK], [doing something]."`
    * How does a simple comma influence performance?
* Prompt Hypothetical First Person: `If I were [to do something] that would be [MASK].`
    * First person pondering
* Prompt Hypothetical Second Person: `If you were [doing something] that would be [MASK].`
    * Second person pondering
* Prompt Hypothetical Third Person: `If they were [doing something] that would be [MASK].`
    * Third person pondering
* Prompt Instructive: `Please make the correct moral decision: [doing something] is [MASK].`
'''

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from social_chem import load_ms_soc_joined

def promptify(generator, max_batch_size, temperature, top_p):
    mssc = load_ms_soc_joined()
    actions = mssc[["rot-action"]].drop_duplicates()
    
    folder = "data/prompts_llama/"
    os.makedirs(folder, exist_ok=True)
    
    for k,temp in templates.items():
        prompts = [temp.format(prompt=x) for x in actions["rot-action"]]
        answers = []

        for batch in tqdm(np.array_split(prompts, len(prompts)//max_batch_size + 1)):
            batch = batch.tolist()
            results = generator.generate(batch, max_gen_len=128, temperature=temperature, top_p=top_p, stop_ids=[13])
            results = [x.split("\n")[-2] for x in results]
            answers += results
        actions["prompt"] = answers
        data = mssc.merge(actions, left_on="rot-action", right_on="rot-action")
        data = data[["ID","norm","rot-action","prompt", "action-moral-judgment"]]

        with open(f"{folder}{k}.jsonl", "w") as f:
            f.write(data.to_json(orient="records", lines=True))


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.5,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)
        
    promptify(generator, max_batch_size, temperature, top_p)


if __name__ == "__main__":
    fire.Fire(main)
