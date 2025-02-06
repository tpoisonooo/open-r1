The puzzles module contains a simple and extensible system for generating and verifying reasoning tasks.
The focus is on tasks where infinite variants can be generated with automatic answer verification, like mathematics, logic puzzles or coding tasks, although
we highly encourage creativity - if you can come up with less STEM-y tasks that can still be rigorously validated, we'd love to see them!

# Generating puzzles

After `pip install`ing the open-r1 repo, you can very quickly get started

```python
>>> from open_r1.puzzles import LinearEquationConfig, LinearEquationTask

>>> config = LinearEquationConfig(seed=123)
>>> task = LinearEquationTask(config)
>>> # Tasks are iterable, so you can iterate with "for task_instance in task:"
>>> task_instance = next(iter(task))
>>> print(task_instance["prompt"])
Solve for a in the following equation:

-9a + 2 = 2
# To score a model output, use task.verify()
>>> task.verify("a = 0", task_instance["ground_truth"])
1.0

>>> # To control the task difficulty, you can use the task's associated config
>>> config = LinearEquationConfig(seed=123)
>>> config.min_coefficient = -1000
>>> config.max_coefficient = 1000
>>> harder_task = LinearEquationTask(config)
```

## Training with puzzles

Training with puzzles is also easy! Puzzles come with a `to_dataset()` method that returns a `Dataset` object. Each
task instance should have a `ground_truth` key that you can use with the task's `verify` method to get a reward
function:

```python
from open_r1.puzzles import LinearEquationTask, LinearEquationConfig
from trl import GRPOConfig, GRPOTrainer

config = LinearEquationConfig(num_tasks=1000)
task = LinearEquationTask(config)
dataset = task.to_dataset()

def reward_func(completions, ground_truth, **kwargs):
    return [task.verify(completion, answer) for completion, answer in zip(completions, ground_truth)]

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

## Adding new puzzles

Adding a new puzzle is easy! You can use the [LinearEquationTask](https://github.com/huggingface/open-r1/blob/main/src/open_r1/puzzles/tasks/math/algebra/linear_equations.py)
as a guide. To add a new task:

1) Create a new task file in the appropriate subdirectory of `open_r1/puzzles/tasks/`
2) In the task file, create a `Task` class that inherits from `BaseTask` and a `Config` class that inherits from `BaseConfig`.
3) For the `Config` class, add any config properties your `Task` is going to use
4) For the `Task` class, set `config_class` to the `Config` class you just created and implement two methods:
    - `generate_sample` should return a dict containing the question and the `ground_truth` answer. Importantly, the task receives `rng` as an argument,
      which is an instance of a NumPy generator class. Any random generations, shuffles, etc. that your class needs to generate task instances
      should use methods on `rng`. You should **never** `import random` and generate your own random numbers, or else reproducibility will be lost!
    - `verify` receives `output` and `answer` as arguments and returns a score between 0 and 1. Some tasks may only return 0. for incorrect answers
      and 1. for correct answers, while others may have scores in between for partially-correct answers. `verify` should be a `@staticmethod`.
5) Finally, when it's all ready, import your task and config in the `__init__.py` file of `open_r1/puzzles`, and add your
   task and config class to `__all__` in that file.

## Coming soon:

- Proper indexing of puzzles
- More puzzle types!
- Lazy loading (if the module gets very big)