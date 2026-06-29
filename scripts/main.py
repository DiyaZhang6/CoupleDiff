import logging
import time
from pathlib import Path


def setup_logging(log_file_name="evaluation.log"):
    """
    Sets up a simple logger to print to both console and a file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_name),
            logging.StreamHandler()
        ]
    )


def simulate_evaluate_on_test_set(test_set_name: str):
    """
    Simulates the evaluation process and returns a dictionary with fixed, predefined metrics.
    This function replaces the real `evaluate_on_test_set`.
    """
    print(f"\nSimulating evaluation on {test_set_name}...")
    # 模拟计算过程的延迟
    for _ in range(5):
        print("Evaluating... [███-----]", end='\r')
        time.sleep(0.2)
        print("Evaluating... [██████--]", end='\r')
        time.sleep(0.2)
    print("Evaluating... [████████] Done. ")

    # --- 预设的、固定的测试结果 ---
    # 您可以在这里修改数值以展示不同的场景
    predefined_results = {
        "Structural Metrics": {
            "Overall RMSD": 2.0123,
            "Ligand RMSD (L-RMSD)": 2.1745,
            "Side-chain RMSD (sc-RMSD)": 0.2904,
        },
        "Success Rates (< 2A)": {
            "Overall Success Rate (%)": 65.30,
            "Ligand Success Rate (%)": 61.60,
            "Side-chain Success Rate (%)": 70.50,
        },
        "Functional (Affinity) Metrics": {
            "PCC": 0.8201,
            "Spearman": 0.8155,
            "RMSE": 1.2033,
            "MAE": 1.0012,
        }
    }

    # 如果是某些特定的数据集，可以模拟没有亲和力指标的情况
    if "PoseBusters" in test_set_name or "ANN" in test_set_name:
        predefined_results["Functional (Affinity) Metrics"] = {}
        # 也可以为这些数据集修改结构指标
        predefined_results["Structural Metrics"]["Overall RMSD"] = 1.7589
        predefined_results["Success Rates (< 2A)"]["Overall Success Rate (%)"] = 72.15

    return predefined_results


def main():
    """
    Main function to simulate and display the testing process with fixed values.
    """
    setup_logging()

    logging.info("--- Starting Evaluation Simulation ---")
    logging.info("Using device: cuda:0")

    # --- 模拟加载模型 ---
    checkpoint_path = Path("checkpoints/train_20250417_112233/best_model_epoch_85.pt")
    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    time.sleep(1)  # 模拟加载时间
    logging.info("Model loaded successfully.")

    # --- 定义要运行的测试集 ---
    test_sets_to_run = ["CASF2016-Drug", "CASF-2016", "PoseBusters_benchmark"]

    for test_set_name in test_sets_to_run:
        logging.info(f"\n{'=' * 20} Evaluating on: {test_set_name} {'=' * 20}")

        # 调用模拟函数获取固定的结果
        results = simulate_evaluate_on_test_set(test_set_name)

        logging.info(f"--- Results for {test_set_name} ---")

        # 打印结构指标
        logging.info("[Structural Metrics]")
        for metric, value in results["Structural Metrics"].items():
            # 使用格式化字符串来对齐输出
            logging.info(f"  {metric:<30}: {value:.4f}")

        # 打印成功率
        logging.info("[Success Rates (< 2Å)]")
        for metric, value in results["Success Rates (< 2A)"].items():
            logging.info(f"  {metric:<30}: {value:.2f}%")

        # 打印功能（亲和力）指标，如果存在
        if results["Functional (Affinity) Metrics"]:
            logging.info("[Functional (Affinity) Metrics]")
            for metric, value in results["Functional (Affinity) Metrics"].items():
                logging.info(f"  {metric:<30}: {value:.4f}")
        else:
            logging.info("[Functional (Affinity) Metrics]")
            logging.info("  (Not available for this dataset)")

    logging.info("\nEvaluation simulation finished.")


if __name__ == '__main__':
    main()