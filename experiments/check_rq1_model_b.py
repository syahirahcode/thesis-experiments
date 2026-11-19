"""Check Model B attribute predictions against the experiment ground truth."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.agents.facial_agent import FacialAgent
from app.core.config import get_settings
from app.core.eval_metrics import load_ground_truth


def main() -> None:
    settings = get_settings()

    # Load ground truth labels
    rows = load_ground_truth(
        settings.GROUND_TRUTH_CSV,
        encoding=settings.GROUND_TRUTH_ENCODING,
    )

    # Load test images
    images = settings.list_test_images()

    if not images:
        print("Tidak ada foto yang ditemukan untuk dievaluasi.")
        return

    ground_truth_by_name = {
        row.file_name.strip(): row for row in rows
    }

    missing = sorted(
        {image.name for image in images}
        - set(ground_truth_by_name)
    )

    if missing:
        raise FileNotFoundError(
            "Ground truth tidak ditemukan untuk foto: "
            + ", ".join(missing)
        )

    # Initialize model
    agent = FacialAgent()

    attributes = ["undertone", "skintone", "saturation"]

    correct = {attribute: 0 for attribute in attributes}
    total = len(images)

    print("\n" + "=" * 110)
    print("EVALUASI PREDIKSI ATRIBUT KULIT — MODEL B")
    print("=" * 110)

    # Analyze every image and compare with its ground truth
    for index, image_path in enumerate(images, start=1):
        expected = ground_truth_by_name[image_path.name].attributes
        predicted = agent.analyze(str(image_path)).attributes

        print(f"\n[{index}/{total}] Foto: {image_path.name}")
        print("-" * 90)

        for attribute in attributes:
            true_label = getattr(expected, attribute)
            predicted_label = getattr(predicted, attribute)

            is_correct = predicted_label == true_label

            if is_correct:
                correct[attribute] += 1

            status = "BENAR" if is_correct else "SALAH"

            print(f"  {attribute.capitalize():12}: "
                  f"Prediksi = {predicted_label} | "
                  f"Label benar = {true_label} | "
                  f"Status = {status}")

    # Print overall accuracy for each attribute
    print("\n" + "=" * 110)
    print("RINGKASAN AKURASI")
    print("=" * 110)
    print(f"Jumlah foto dievaluasi: {total}")

    for attribute in attributes:
        count = correct[attribute]
        accuracy = 100 * count / total if total else 0.0

        print(
            f"{attribute.capitalize():12}: "
            f"{count}/{total} benar "
            f"({accuracy:.2f}%)"
        )

    # Calculate accuracy across all three attributes
    total_predictions = total * len(attributes)
    total_correct = sum(correct.values())

    overall_accuracy = (
        100 * total_correct / total_predictions
        if total_predictions
        else 0.0
    )

    print("-" * 50)
    print(
        f"Akurasi keseluruhan: "
        f"{total_correct}/{total_predictions} "
        f"({overall_accuracy:.2f}%)"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
