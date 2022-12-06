
def prefetch_loader(loader, device):
    result = []
    for data in loader:
        items = []
        for item in data:
            items.append(item.to(device))
        result.append(tuple(items))
    return result