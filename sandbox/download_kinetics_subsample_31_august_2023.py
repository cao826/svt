import torchvision

# Change these paths to where you want to save the dataset
download_dir = "./kinetics_data"
subset_dir = "./kinetics_subset"

# Download the dataset metadata
dataset = torchvision.datasets.Kinetics400(
    download_dir, frames_per_clip=1, num_workers=1, max_len=None
)

# Select a subset of the dataset
subset_indices = [0, 1, 2]  # Replace with the indices you want
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

# Create a loader for the subset dataset
loader = torch.utils.data.DataLoader(subset_dataset, batch_size=1, shuffle=True)

# Download and iterate through the subset videos
for batch in loader:
    video, label = batch
    print(f"Video shape: {video.shape}, Label: {label.item()}")
