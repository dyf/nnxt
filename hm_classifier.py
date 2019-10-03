import torch
import numpy as np
import pandas as pd

class HMDataSet(torch.utils.data.Dataset):
    def __init__(self, reads_matrix, metadata):
        self.reads_matrix = reads_matrix.astype(np.float32)
        self.metadata = metadata
    
        self.cat_names = [
            'mouse', 'human', 'cells', 'nuclei', 
            #'male', 'female', 
            #'lh', 'rh'
        ]
        self.cats = np.array([ 
            (metadata.organism == 'Mus musculus').values,
            (metadata.organism == 'Homo sapiens').values,
            (metadata.sample_type == 'Cells').values,
            (metadata.sample_type == 'Nuclei').values,
            #(metadata.sex == "M").values,
            #(metadata.sex == "F").values,
            #(metadata.brain_hemisphere == "L").values,
            #(metadata.brain_hemisphere == "R").values
        ]).T.astype(np.float32)
    
    def __len__(self):
        return self.metadata.shape[0]
    
    def __getitem__(self, idx):
        return self.reads_matrix.iloc[idx].values, self.cats[idx]

if __name__ == "__main__":
    torch.set_num_threads(8)
    
    use_cuda = False#torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    sample_metadata = pd.read_csv('classifier_sample_metadata.csv', index_col='sample_id')
    sample_metadata.sort_index(inplace=True)
    sample_metadata.fillna(value=0.0, inplace=True)
    
    print(sample_metadata.columns)
    
    reads_matrix = pd.read_hdf('classifier_reads_matrix.h5', key='matrix', index_col='sample_id')
    reads_matrix.sort_index(inplace=True)
    
    full_ds = HMDataSet(reads_matrix, sample_metadata)

    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size, test_size])

    params = {
       'batch_size': 50,
       'shuffle': True,
       'num_workers': 0
    }

    train_generator = torch.utils.data.DataLoader(train_ds, **params)
    test_generator = torch.utils.data.DataLoader(test_ds, batch_size=test_size)

    for test_input, test_labels in test_generator:
        print(f'{test_labels.sum()}/{len(test_labels)}')
    
    n_genes = reads_matrix.shape[1]
    
    model = torch.nn.Sequential(
        torch.nn.Dropout(.3),
        torch.nn.Linear(n_genes, 30),        
        torch.nn.ReLU(),
        torch.nn.Linear(30, 10),        
        torch.nn.ReLU(),
        torch.nn.Linear(10, len(full_ds.cat_names)),
        torch.nn.Sigmoid()
    )
    #model.cuda()

    #criterion = torch.nn.MultiLabelSoftMarginLoss()
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()
    #criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1)
    
    n_epochs = 500
    #loss_history = []
    #test_loss_history = []
    for ei in range(n_epochs):
        # Training
        for bi, (local_batch, local_labels) in enumerate(train_generator):
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            
            def closure():
                optimizer.zero_grad()
                output = model(local_batch)
                loss = criterion(output, local_labels)                
                loss.backward()                
                return loss
            
            loss = optimizer.step(closure)  
            #scheduler.step()
            
            if bi % 100 == 0:
                print(f'Epoch: [{ei+1}/{n_epochs}], Batch: {bi}, Loss: {loss:.8f}')
            #loss_history.append(loss)
        
        for test_input, test_labels in test_generator:
            test_out = model.forward(test_input)
            test_loss = criterion(test_out, test_labels)
            print(f'Test Loss: {test_loss:.8f}')
            #test_loss_history.append(test_loss)
            
            out_labels = np.round(test_out.detach().numpy())
            print(out_labels.shape, test_labels.shape)
            for i, cat_name in zip(range(out_labels.shape[1]), full_ds.cat_names):
                matching_labels = (test_labels.numpy()[:,i] == out_labels[:,i]).sum()
                print(f'Category {cat_name} matches {matching_labels}/{test_size}')
                
                
            
        torch.save(model.state_dict(), "hmclassifier.weights")