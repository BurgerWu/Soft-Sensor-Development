import numpy as np
def KS_algorithm(dataset_x, dataset_y, train_size = 0.7, return_dataset = True):
    """
    This function performs Kennard-Stone sample selection method.  
    
    Input: 
    - dataset_x: Input dataset for x
    - dataset_y: Input dataset for y
    - train_szie: Size of train set, either between 0 to 1 by ratio, or 2 to total number of samples by quantity
    - return_dataset: If True, return train_x, test_x, train_y, test_y as train_test_split does
                      If False, return index of chosen x
    
    Output: 
    - If return_dataset is True, return train_x, test_x, train_y, test_y as train_test_split does
    - If return_dataset is False, return index of chosen x
    """
    #Decide number of total samples
    n = dataset_x.shape[0]
    
    #Turn into train_size to float
    try: 
        train_size = float(train_size)
    except Expection as e:
        print(e)
    
    #Jusge if the train_size is by ratio or quantity
    if 0 < train_size < 1: 
        num_of_samples = round(n*train_size)
    elif 1 < train_size < 2: 
        return "Train size must be between 1 to 0 or 2 to number of total samples"       
    elif train_size == n or train_size == 1:
        return [x for x in range(n)]
    elif 2 <= train_size < n:
        num_of_samples = round(train_size)
    elif train_size > n:
        return "Num_of_samples must be less than or equal to original number of samples"
    
    #Create original sample index
    samples = [x for x in range(n)]
    
    #Create distance matrix
    print('Building distance matrix')
    dist_matrix = create_dist_matrix(dataset_x)

    #Pick first two furthest away data points
    print('Picking two furthest away samples')
    max_array = np.argmax(dist_matrix == np.max(dist_matrix), axis = 1)
    ind1 = max_array[max_array != 0][0]
    ind2 = max_array[ind1]

    #Initiate chosen sample list, append first two samples and remove from remaining sample list
    chosen_samples = [ind1, ind2]
    samples.remove(ind1)
    samples.remove(ind2)    
    remaining_samples = num_of_samples - 2

    #Continue picking samples if not yet reaching target number of samples
    while remaining_samples > 0:
        sub_array = np.min(dist_matrix[:,chosen_samples], axis=1)
        chosen = np.argmax(sub_array)
        chosen_samples.append(chosen)
        samples.remove(chosen)
        remaining_samples -= 1
    
    #See if this function should return train and test set or simply index of chosen samples
    if return_dataset == True:
        not_chosen_samples = [x for x in range(n) if x not in chosen_samples]
        return dataset_x.iloc[chosen_samples,:], dataset_x.iloc[not_chosen_samples,:], dataset_y.iloc[chosen_samples,:], dataset_y.iloc[not_chosen_samples,:]   
    else:
        return chosen_samples



def create_dist_matrix(matrix):
    """
    This function creates distance matrix based on Euclidean distance between each data points 
    
    Input: Data matrix
    Output: Distance matrix
    """
    #Convert input data matrix to numpy matrix
    matrix = np.array(matrix)
    n = matrix.shape[0]
    
    #Iterate through number of samples to create distance matrix
    for i in range(n):
        dist_array = euclidean_distance(matrix[i,:], matrix)
        if i == 0:
            dist_matrix = dist_array
        else:
            dist_matrix = np.concatenate((dist_matrix, dist_array), axis = 1)
    return dist_matrix


def euclidean_distance(data1, data2):
    """
    This function calculates Euclidean distance between data1 and data2
    
    Input: Two data points or array
    Output: Distance or distance array
    """
    #Convert data into numpy array
    array1 = np.array(data1)
    array2 = np.array(data2)
    
    #Create distance array
    dist_array = np.sqrt(np.sum((array2-array1)**2, axis=1))
    
    #Reshape array before return results
    return np.reshape(dist_array, [len(dist_array),1])

