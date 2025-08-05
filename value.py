# Data provided by the user
branches = [
    {
      "branch_id": 0,
      "step_count": 5,
      "score": 0.0016002703562340967,
      "uncertainty": 0.68359375,
      "value": 0.62890625,
      "total_tokens": 393,
      "extracted_answer": "H"
    },
    {
      "branch_id": 1,
      "step_count": 5,
      "score": 0.0009111576212471131,
      "uncertainty": 0.68359375,
      "value": 0.39453125,
      "total_tokens": 433,
      "extracted_answer": "F"
    },
    {
      "branch_id": 2,
      "step_count": 5,
      "score": 0.0006810059274755928,
      "uncertainty": 0.458984375,
      "value": 0.48828125,
      "total_tokens": 717,
      "extracted_answer": "H"
    },
    {
      "branch_id": 3,
      "step_count": 6,
      "score": 0.0016875537865748708,
      "uncertainty": 0.5703125,
      "value": 0.98046875,
      "total_tokens": 581,
      "extracted_answer": "H"
    },
    {
      "branch_id": 4,
      "step_count": 6,
      "score": 0.0008914421553090333,
      "uncertainty": 0.484375,
      "value": 0.5625,
      "total_tokens": 631,
      "extracted_answer": "F"
    },
    {
      "branch_id": 5,
      "step_count": 6,
      "score": 0.0008198302469135802,
      "uncertainty": 0.69140625,
      "value": 0.53125,
      "total_tokens": 648,
      "extracted_answer": "C"
    },
    {
      "branch_id": 6,
      "step_count": 6,
      "score": 0.0012683640438247012,
      "uncertainty": 0.69140625,
      "value": 0.63671875,
      "total_tokens": 502,
      "extracted_answer": "H"
    },
    {
      "branch_id": 7,
      "step_count": 7,
      "score": 0.0009672024760383386,
      "uncertainty": 0.48828125,
      "value": 0.60546875,
      "total_tokens": 626,
      "extracted_answer": "A"
    },
    {
      "branch_id": 8,
      "step_count": 7,
      "score": 0.0008157869664634146,
      "uncertainty": 0.48828125,
      "value": 0.53515625,
      "total_tokens": 656,
      "extracted_answer": "C"
    },
    {
      "branch_id": 9,
      "step_count": 7,
      "score": 0.0008566810344827586,
      "uncertainty": 0.55859375,
      "value": 0.62109375,
      "total_tokens": 725,
      "extracted_answer": "G"
    },
    {
      "branch_id": 10,
      "step_count": 8,
      "score": 0.00128125,
      "uncertainty": 0.58984375,
      "value": 0.9609375,
      "total_tokens": 750,
      "extracted_answer": "H"
    },
    {
      "branch_id": 11,
      "step_count": 8,
      "score": 0.0014292547376093295,
      "uncertainty": 0.58984375,
      "value": 0.98046875,
      "total_tokens": 686,
      "extracted_answer": "F"
    },
    {
      "branch_id": 12,
      "step_count": 8,
      "score": 0.0014366451367781156,
      "uncertainty": 0.51171875,
      "value": 0.9453125,
      "total_tokens": 658,
      "extracted_answer": "G"
    },
    {
      "branch_id": 13,
      "step_count": 8,
      "score": 0.001025663407821229,
      "uncertainty": 0.51171875,
      "value": 0.734375,
      "total_tokens": 716,
      "extracted_answer": "H"
    },
    {
      "branch_id": 14,
      "step_count": 8,
      "score": 0.0008309014267185473,
      "uncertainty": 0.345703125,
      "value": 0.640625,
      "total_tokens": 771,
      "extracted_answer": "G"
    },
    {
      "branch_id": 15,
      "step_count": 9,
      "score": 0.001375679347826087,
      "uncertainty": 0.58984375,
      "value": 0.94921875,
      "total_tokens": 690,
      "extracted_answer": "F"
    },
    {
      "branch_id": 16,
      "step_count": 9,
      "score": 0.0010775862068965517,
      "uncertainty": 0.58984375,
      "value": 0.875,
      "total_tokens": 812,
      "extracted_answer": "C"
    },
    {
      "branch_id": 17,
      "step_count": 9,
      "score": 0.0013077445652173913,
      "uncertainty": 0.57421875,
      "value": 0.90234375,
      "total_tokens": 690,
      "extracted_answer": "F"
    }
]

# Separate the values based on the "extracted_answer"
g_values = []
other_values = []

for branch in branches:
    if branch['extracted_answer'] == 'G':
        g_values.append(branch['value'])
    else:
        other_values.append(branch['value'])

# Calculate the average for each group
# Check if the lists are not empty to avoid division by zero
average_g = sum(g_values) / len(g_values) if g_values else 0
average_other = sum(other_values) / len(other_values) if other_values else 0

# Print the results
print("Average 'value' for branches with 'G' as the final answer:")
print(average_g)
print("\nAverage 'value' for branches with other letters as the final answer:")
print(average_other)
print("Top-4 Branches' Values")
branches.sort(key=lambda b: b["value"], reverse=True)
print([b["extracted_answer"] for b in branches[:4]])