import pandas as pd
class FeatureEngineeringPipeline:
    def __init__(self):
        self.steps = []
    def add_step(self, name, transformer):
        self.steps.append((name, transformer))
    def remove_step(self, name):
        self.steps = [
            (step_name, transformer)
            for step_name, transformer in self.steps
            if step_name != name
        ]
    def show_steps(self):
        print("Pipeline Steps")
        for i, (name, _) in enumerate(self.steps, start=1):
            print(f"{i}. {name}")
    def transform(self, df):
        result = df.copy()
        print("Pipeline")
        print(f"Initial Shape: {result.shape}")
        for name, transformer in self.steps:
            print(f"\nRunning Step: {name}")
            old_shape = result.shape
            result = transformer(result)
            print(
                f"Shape Changed: "
                f"{old_shape} -> {result.shape}"
            )
        print("Pipeline Complete")
        print(f"Final Shape: {result.shape}")
        return result
    def fit_transform(self, df):
        return self.transform(df)