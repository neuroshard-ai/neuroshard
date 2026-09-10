"""Set the genesis numerical profile before importing the neural runtime."""
import os


def configure():
    os.environ.update(ATEN_CPU_CAPABILITY="default", MKL_ENABLE_INSTRUCTIONS="SSE4_2",
                      OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def chain():
    configure()
    from neuroshard.publicnet.cli import main
    main()


def worker():
    configure()
    from neuroshard.publicnet.worker_cli import main
    main()
