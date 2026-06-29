loader = get_data_loader(config, "train", config["training"])
batch = next(iter(loader))

print(batch["backbone"].x.shape, batch["backbone"].pos.shape)
print(batch["sidechain"].x.shape, batch["sidechain"].pos.shape)
print(batch["drug"].x.shape, batch["drug"].pos.shape)
print(batch.r_true.shape)