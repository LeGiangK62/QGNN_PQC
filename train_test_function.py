import torch
import torch.nn as nn

criterion = nn.MSELoss()

def train(model, train_loader, optimizer, setting='supervised'):
    model.train()
    total_loss = 0
    total_rate = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for channel_matrices, weight_sumrates, _, labels in train_loader:
        channel_matrices = channel_matrices[0].T.to(device).to(torch.float32)
        labels = labels[0].to(device).to(torch.float32)
        weight_sumrates = weight_sumrates[0].to(device).to(torch.float32)

        # original_channel = channel_matrices.clone()
        # channel_matrices = (channel_matrices - channel_matrices.min()) / (channel_matrices.max() - channel_matrices.min())

        optimizer.zero_grad()

        output = model(channel_matrices, weight_sumrates).float()
        if setting == 'supervised':
            loss = criterion(output, labels)
        elif setting == 'unsupervised':
            loss = sum_weighted_rate(channel_matrices, output, weight_sumrates, var)
            loss = torch.neg(loss)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            power_new = output #quantize_output(output)
            sum_rate = sum_weighted_rate(channel_matrices, power_new, weight_sumrates, var)
        total_rate += sum_rate.item()
        total_loss += loss.item()

    return total_loss / len(train_loader), total_rate / len(train_loader)


def test(model, test_loader):
    model.eval()
    total_loss = 0
    total_rate = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for channel_matrices, weight_sumrates, _, labels in test_loader:
            channel_matrices = channel_matrices[0].T.to(device).to(torch.float32)
            labels = labels[0].to(device).to(torch.float32)
            weight_sumrates = weight_sumrates[0].to(device).to(torch.float32)
            # original_channel = channel_matrices.clone()
            # channel_matrices = (channel_matrices - channel_matrices.min()) / (channel_matrices.max() - channel_matrices.min())

            output = model(channel_matrices, weight_sumrates).float()
            power_new = output #quantize_output(output)
            sum_rate = sum_weighted_rate(channel_matrices, power_new, weight_sumrates, var)
            total_rate += sum_rate.item()

    return total_rate / len(test_loader)


def quantize_output(output, num_levels=4):
    levels = torch.linspace(0, 1, steps=num_levels, device=output.device)
    # Find closest level for each output value
    quantized_output = torch.zeros_like(output)
    for i in range(output.shape[0]):
        quantized_output[i] = levels[torch.argmin(torch.abs(levels - output[i]))]
    return quantized_output


def sum_weighted_rate(h, p, w, n0):

    all_signal = torch.square(h * p.view(-1, 1))
    des_signal = torch.diag(all_signal)
    rx_signal = torch.sum(all_signal, dim=0)
    inteference = rx_signal - des_signal + n0

    sinr = des_signal/inteference
    w_sumrate = torch.log2(1 + sinr * w)
    return torch.sum(w_sumrate)