import csv

def format_csv(input_filename, output_filename):
  """
  Formats a CSV file containing body pose data.

  Args:
    input_filename: Path to the input CSV file.
    output_filename: Path to the output CSV file.
  """

  with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header row
    writer.writerow([
        "left_shoulder_x", "left_shoulder_y", "right_shoulder_x", "right_shoulder_y",
        "left_hip_x", "left_hip_y", "right_hip_x", "right_hip_y",
        "left_knee_x", "left_knee_y", "right_knee_x", "right_knee_y",
        "left_ankle_x", "left_ankle_y", "right_ankle_x", "right_ankle_y",
        "label"
    ])

    # Skip the header row in the input file
    next(reader)

    for row in reader:
      label = row[0]
      # Remap labels
      if label == "C":
        label = 0
      elif label == "L":
        label = 1
      elif label == "H":
        label = 2

      # Extract relevant columns
      new_row = [
          row[5], row[6],  # left_shoulder_x, left_shoulder_y
          row[8], row[9],  # right_shoulder_x, right_shoulder_y
          row[29], row[30], # left_hip_x, left_hip_y
          row[32], row[33], # right_hip_x, right_hip_y
          row[35], row[36], # left_knee_x, left_knee_y
          row[38], row[39], # right_knee_x, right_knee_y
          row[41], row[42], # left_ankle_x, left_ankle_y
          row[44], row[45], # right_ankle_x, right_ankle_y
          label
      ]
      writer.writerow(new_row)

# Example usage:
input_csv_file = '/Users/defeee/Downloads/train.csv'  # Replace with your input filename
output_csv_file = 'output.csv'  # Replace with desired output filename
format_csv(input_csv_file, output_csv_file)