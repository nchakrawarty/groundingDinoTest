def train_one_epoch(model, dataloader, optimizer, epoch, device):
    model.train()
    for batch_idx, (images, bboxes, labels) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        bboxes = [bbox.to(device) for bbox in bboxes]
        labels = [label.to(device) for label in labels]

        # Forward pass
        outputs = model(images, bboxes, labels)
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, bboxes, labels) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            bboxes = [bbox.to(device) for bbox in bboxes]
            labels = [label.to(device) for label in labels]

            # Forward pass
            outputs = model(images, bboxes, labels)
            print(f"Evaluation - Batch {batch_idx + 1}/{len(dataloader)}")
