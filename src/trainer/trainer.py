import torchmetrics
import torch
import segmentation_models_pytorch as smp


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        # print(output.shape, data.shape)
        # print(output.device)
        # print(target.device)
        # Ensure the output is in float32 (e.g., for log_softmax and other floating-point operations)
        output = output.to(torch.float32)

        # For the target, it should be in long format (class indices)
        target = target.to(torch.long)

        # Compute the loss
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    eval_loss = 0.0
    dice_score = 0.0
    dice_score_function = torchmetrics.Dice(num_classes=13, average='micro').to(device)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            # If the model outputs logits, apply softmax
            output = torch.softmax(output, dim=1)  # Apply softmax over the class dimension

            loss = criterion(output, target.long())
            eval_loss += loss.item()

            # Update dice score calculation
            dice_score += dice_score_function(output, target.long())

    return eval_loss / len(dataloader), dice_score / len(dataloader)


def train(model, train_datasets, test_datasets, num_epochs=200):
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=8, shuffle=True, num_workers=2)
    eval_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=8, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model = model.to(device)
    criterion = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  
    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)

        # Evaluation
        eval_loss, dice_score = evaluate(model, eval_dataloader, criterion, device)

        # Print results
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Eval Loss = {eval_loss:.4f}, Dice Score = {dice_score:.4f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), 'best.pt')
