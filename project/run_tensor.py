"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import argparse
import random
import time

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # ASSIGN2.5
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()
        # END ASSIGN2.5


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # ASSIGN2.5
        batch, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch, in_size, 1)
        ).sum(1).view(batch, self.out_size) + self.bias.value.view(self.out_size)
        # END ASSIGN2.5


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        self.epoch_times = []
        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.perf_counter()
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()
            self.epoch_times.append(time.perf_counter() - epoch_start)

            # Logging
            if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)
                if log_fn is default_log_fn:
                    print(
                        f"Time per epoch: {self.epoch_times[-1]:.4f}s; "
                        f"mean: {sum(self.epoch_times) / len(self.epoch_times):.4f}s"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*minitorch.datasets, "all"], default="Simple")
    parser.add_argument("--points", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=2)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = list(minitorch.datasets) if args.dataset == "all" else [args.dataset]
    for name in names:
        random.seed(args.seed)
        data = minitorch.datasets[name](args.points)
        print(f"Dataset: {name}", flush=True)
        trainer = TensorTrain(args.hidden)
        trainer.train(data, args.rate, max_epochs=args.epochs)
        predictions = trainer.run_many(data.X).view(data.N)
        correct = sum((predictions[i] > 0.5) == data.y[i] for i in range(data.N))
        print(f"Final accuracy after update: {correct}/{data.N}", flush=True)
