def show_ram_usage(label="Speicherstatus"):
    """
    Zeigt den aktuellen RAM-Verbrauch an (mehrfach aufrufbar)
    Korrigierte Version, die den tatsächlichen Speicherverbrauch anzeigt

    Args:
        label (str): Beschreibender Text für diesen Aufruf

    Returns:
        dict: Dictionary mit allen Speichermesswerten
    """
    import psutil
    import os
    import torch
    import gc
    import time

    print("\n" + "="*120)
    print(f"RAM-NUTZUNG: {label}".center(120))
    print("="*120)
    print(f"Zeitstempel: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*120)

    # 1. Systemweiter Speicherverbrauch des Prozesses (RSS)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 ** 2)  # Resident Set Size in MiB

    # 2. PyTorch-Speicher (CPU + GPU)
    cpu_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    cpu_mem_mb = cpu_mem / (1024 ** 2)

    gpu_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    gpu_mem_mb = gpu_mem / (1024 ** 2)

    # 3. Python-Objekte (tracemalloc)
    import tracemalloc
    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()
    py_mem_mb = sum(stat.size for stat in snapshot.statistics('lineno')) / (1024 ** 2)

    # 4. Speicher aller PyTorch-Tensoren (inkl. der großen fMRI-Volumes)
    total_tensor_mem = 0
    tensor_count = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            total_tensor_mem += obj.element_size() * obj.nelement()
            tensor_count += 1

    tensor_mem_mb = total_tensor_mem / (1024 ** 2)

    # 5. Berechnung des tatsächlichen Speicherverbrauchs
    # (RSS sollte den größten Teil abdecken, da die fMRI-Volumes hier gespeichert sind)
    print(f"System-Prozess-Speicher (RSS): {rss_mb:.2f} MiB")
    print(f"PyTorch CPU-Speicher: {cpu_mem_mb:.2f} MiB")
    print(f"PyTorch GPU-Speicher: {gpu_mem_mb:.2f} MiB")
    print(f"Python-Objekte (tracemalloc): {py_mem_mb:.2f} MiB")
    print(f"PyTorch-Tensoren: {tensor_mem_mb:.2f} MiB ({tensor_count} Tensoren)")
    print(f"GESAMT (RSS): {rss_mb:.2f} MiB")

    # Berechnung des erwarteten Speichers für deine fMRI-Volumes

    print("-"*120)

    # Starte tracemalloc neu für den nächsten Aufruf
    tracemalloc.start()

    return {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "rss_mb": rss_mb,
        "cpu_mem_mb": cpu_mem_mb,
        "gpu_mem_mb": gpu_mem_mb,
        "py_mem_mb": py_mem_mb,
        "tensor_mem_mb": tensor_mem_mb,
        "tensor_count": tensor_count
    }