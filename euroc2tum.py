import csv
import argparse

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Convert EuRoC CSV groundtruth to TUM format.")
parser.add_argument("input_file", help="Path to the EuRoC data.csv file")
parser.add_argument("output_file", help="Path to save the converted TUM file")
args = parser.parse_args()

# --- File conversion ---
with open(args.input_file, "r") as f_in, open(args.output_file, "w") as f_out:
    # Read and normalize header
    header = f_in.readline().strip()
    fieldnames = [h.strip() for h in header.split(",")]

    reader = csv.DictReader(f_in, fieldnames=fieldnames)

    for row in reader:
        # Skip empty or commented lines
        if not row[fieldnames[0]] or row[fieldnames[0]].startswith("#"):
            continue

        try:
            timestamp = int(row[fieldnames[0]])  # integer timestamp only
            tx = float(row["p_RS_R_x [m]"])
            ty = float(row["p_RS_R_y [m]"])
            tz = float(row["p_RS_R_z [m]"])
            qw = float(row["q_RS_w []"])
            qx = float(row["q_RS_x []"])
            qy = float(row["q_RS_y []"])
            qz = float(row["q_RS_z []"])

            # Write in TUM format: timestamp tx ty tz qx qy qz qw
            f_out.write(f"{timestamp} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
        except KeyError as e:
            print(f"⚠️ Missing field in CSV: {e}")
            break

print(f"✅ Saved converted file to:\n{args.output_file}")