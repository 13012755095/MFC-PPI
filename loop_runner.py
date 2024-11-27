import subprocess

ppi_path = './protein_info/protein.actions.SHS27k.STRING.pro2.txt'
pseq = './protein_info/protein.SHS27k.sequences.dictionary.pro3.tsv'
# ppi_path = './protein_info/protein.actions.SHS148k.txt'
# pseq = './protein_info/protein.SHS148k.sequences.dictionary.tsv'
split = 'random'

p_feat_matrix = './protein_info/protein.nodes.SHS27k.D12.pt'
p_adj_matrix = './protein_info/protein.rball.edges.SHS27kD12.npy'
# p_feat_matrix = './protein_info/protein.nodes.SHS148k.pt'
# p_adj_matrix = './protein_info/protein.rball.edges.SHS148k.npy'
save_path = './result_save'
epoch_num = 300
working_directory = '/Your project directory'  # Your project directory

conda_env_python = '/Change to your virtual environment path'  # Change to your virtual environment path

# Sets the number of times to execute
num_iterations = 1

for i in range(num_iterations):
    print(f"Running iteration {i + 1}/{num_iterations}")
    command = [
        conda_env_python, "model_train.py",
        "--ppi_path", ppi_path,
        "--pseq", pseq,
        "--split", split,
        "--p_feat_matrix", p_feat_matrix,
        "--p_adj_matrix", p_adj_matrix,
        "--save_path", save_path,
        "--epoch_num", str(epoch_num)
    ]

    process = subprocess.Popen(command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    rc = process.poll()
    if rc != 0:
        error_output = process.stderr.read()
        print(f"Error occurred: {error_output}")
