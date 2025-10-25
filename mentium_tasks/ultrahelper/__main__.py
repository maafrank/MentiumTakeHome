from .load import load_trainer, load_model, load_deployment_model, load_postprocessor
from .nn import InputSignatureWrap
from torch.fx import symbolic_trace
import argparse
import torch
import threading
import queue
import time

def get_args():
    parser = argparse.ArgumentParser(description="Pipeline control flags")
    parser.add_argument('--train', action='store_true', help='Runs training')
    parser.add_argument('--trace', action='store_true', help='Traces the modified model')
    parser.add_argument('--pipeline', action='store_true', help='Execute full parallel pipeline on infinite loop')
    return parser.parse_args()


def run_parallel_pipeline():
    """
    Parallel inference pipeline with GPU hardware model and CPU postprocessor.
    Runs in an infinite loop displaying FPS and latency metrics.
    """
    print("=" * 60)
    print("Starting Parallel Inference Pipeline")
    print("=" * 60)

    # Load models
    print("\nLoading deployment model (for GPU)...")
    deployment_model = load_deployment_model()
    deployment_model.eval()

    print("Loading postprocessor (for CPU)...")
    postprocessor = load_postprocessor()
    postprocessor.eval()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nHardware device: {device}")
    deployment_model = deployment_model.to(device)

    # Communication queues between threads
    feature_queue = queue.Queue(maxsize=2)  # Small buffer to keep threads in sync

    # Metrics tracking
    frame_times = []
    hw_latencies = []
    post_latencies = []

    # Create dummy input (simulating camera frames)
    dummy_input = torch.randn(1, 3, 640, 640)

    def hardware_thread():
        """Thread running deployment model on GPU."""
        input_tensor = dummy_input.to(device)
        while True:
            start = time.time()
            with torch.no_grad():
                feats = deployment_model(input_tensor)
            hw_time = time.time() - start

            # Transfer to CPU
            feats_cpu = [(b.cpu(), c.cpu(), k.cpu()) for b, c, k in feats]

            # Send to postprocessor
            feature_queue.put((feats_cpu, hw_time))

    def postprocessor_thread():
        """Thread running postprocessor on CPU."""
        while True:
            feats_cpu, hw_time = feature_queue.get()

            start = time.time()
            with torch.no_grad():
                predictions = postprocessor(feats_cpu)
            post_time = time.time() - start

            # Record metrics
            hw_latencies.append(hw_time)
            post_latencies.append(post_time)
            frame_times.append(time.time())

            # Keep only recent metrics (last 100 frames)
            if len(frame_times) > 100:
                frame_times.pop(0)
                hw_latencies.pop(0)
                post_latencies.pop(0)

    # Start threads
    print("\nStarting parallel threads...")
    hw_thread = threading.Thread(target=hardware_thread, daemon=True)
    post_thread = threading.Thread(target=postprocessor_thread, daemon=True)

    hw_thread.start()
    post_thread.start()

    print("\n" + "=" * 60)
    print("Pipeline running - Press Ctrl+C to stop")
    print("=" * 60)

    # Display metrics loop
    try:
        while True:
            time.sleep(1.0)  # Update metrics every second

            if len(frame_times) >= 2:
                # Calculate FPS from recent frames
                time_span = frame_times[-1] - frame_times[0]
                fps = len(frame_times) / time_span if time_span > 0 else 0

                # Calculate average latencies
                avg_hw_latency = sum(hw_latencies) / len(hw_latencies) * 1000  # ms
                avg_post_latency = sum(post_latencies) / len(post_latencies) * 1000  # ms
                total_latency = avg_hw_latency + avg_post_latency

                # Display metrics
                print(f"\r📊 FPS: {fps:6.2f} | "
                      f"Hardware: {avg_hw_latency:6.2f}ms | "
                      f"Postproc: {avg_post_latency:6.2f}ms | "
                      f"Total: {total_latency:6.2f}ms",
                      end='', flush=True)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Pipeline stopped by user")
        print("=" * 60)


def main():
    args = get_args()
    if args.train:
        trainer = load_trainer()
        trainer.train()
    elif args.trace:
        model = load_model()
        model = InputSignatureWrap(model)
        model_traced = symbolic_trace(model)
        model_traced.graph.print_tabular()
    elif args.pipeline:
        run_parallel_pipeline()

if __name__ == '__main__':
    main()
