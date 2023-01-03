def combine(combination_name, models, priors, weights, dataloader, device):
    """
    INPUTS
    combination_name: name of the combination as a string choosing between different combinations and solvers
    models: list of models used as source entities
    priors: list of priors in dimension (source_domain+1,  n_classes) where +1 is the target domain.
    weights: list of weights in dimension (source_domain+1, n_classes, n_composantes)  where +1 is the target domain.
    OUTPUT:
    confusion_matrix
    metrics 


    !!! Does not work with a batch_size greater than 1 !!! 
    """
    
    if priors == None:
        priors = torch.ones((weights.shape[0], weights.shape[1]))/weights.shape[1]

    for model in models:
        model.eval()

    if "bayesian" in combination_name:

        alphas = priors.unsqueeze(-1).repeat(1,1,weights.shape[-1])*weights # (source_domain+1, n_classes, n_composantes)
        betas = alphas.sum(dim=1) # (source_domain+1, n_composantes)

    # Initialize the confusion matrix
    evaluator = Evaluator(models[0].num_classes)
    with torch.no_grad():

        with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
            for i, (images, labels, image_weight) in t: 


                images = images.to(device)
                labels = labels.to(device)

                outputs = list()

                for model in models:

                    outputs.append(model(images)[0])

                outputs = torch.stack(outputs)

                if dataloader.dataset.calibrated:
                    outputs = calibrate_posteriors(outputs,dataloader.dataset.priors.cuda())

                # Outputs has shape (n_models, n_classes, height, width)


                if "naive" in combination_name:
                    outputs = solve_naive(outputs, dataloader.dataset.percent, combination_name)

                elif "bayesian" in combination_name:
                    outputs = solve_bayesian(outputs, alphas, betas, weights[-1], priors[-1]).unsqueeze(0)
                elif "posterior" in combination_name:
                    np.save('posteriors/CityScapes_{}'.format(i), outputs[0].cpu().detach().numpy())
                    np.save('posteriors/BDD100K_{}'.format(i), outputs[1].cpu().detach().numpy())
                    outputs = outputs[0].unsqueeze(0)

                #Output should be (n_classes, height, width) got which the elemnts are the posteriors
                evaluator.update(labels.cpu().numpy(),outputs.cpu().detach().numpy(), image_weight[0].item())

        accuracy = evaluator.compute_accuracy()
        miou = evaluator.compute_miou()
        accuracy_balanced = evaluator.compute_accuracy(balanced = True)
        miou_balanced = evaluator.compute_miou(balanced = True)

        logging.info("Performance at combination: ")

        logging.info("Performance with method: " + combination_name)

        logging.info("accuracy: " +  str(accuracy))
        logging.info("miou: " +  str( miou))
        logging.info("accuracy_balanced: " +  str(accuracy_balanced))
        logging.info("miou_balanced: " +  str( miou_balanced))

        return accuracy, miou, accuracy_balanced, miou_balanced, evaluator

def solve_naive(outputs, percent, combination_name):

    if combination_name == "naive_log_softmax":
        outputs[0] = torch.nn.LogSoftmax(0)(outputs[0])
        outputs[1] = torch.nn.LogSoftmax(0)(outputs[1])
    elif combination_name == "naive_softmax":
        outputs[0] = torch.nn.Softmax(0)(outputs[0])
        outputs[1] = torch.nn.Softmax(0)(outputs[1])
    
    return ((percent/100.)*outputs[0] + (1 - (percent/100.))*outputs[1]).unsqueeze(0)

def solve_bayesian(outputs, alphas, betas, weights, priors):
    """
    outputs dim: (source_domain, n_classes, height, width)
    alphas dim: (source_domain+1, n_classes, n_composantes)
    betas dim: (source_domain+1, n_composantes)
    weights dim (n_classes, n_composantes) the weight of the target domain
    priors dim (n_classes) the priors of the target domain
    
    return outputs_target dim:(n_classes, height, width)
    """
    
    n_classes, height, width = outputs.shape[1], outputs.shape[2], outputs.shape[3]
    outputs = torch.nn.functional.softmax(torch.transpose(torch.transpose(torch.flatten(outputs,-2,-1),0,2),1,2), dim=2)
    # Outputs is (n_pixels, n_sources, n_classes) with n_pixels = height*width
    outputs_target = []
    
    alphas_tmp = torch.flatten(alphas[:-1],0,1) # (n_sources*n_classes, n_composantes)
    betas_tmp = torch.flatten(betas[:-1].unsqueeze(1).repeat(1,alphas.shape[1],1),0,1) # flatten(n_sources, n_classes, n_composantes) -> (n_sources*n_classes, n_composantes)

      # A is n_equations (n_models*n_classes), n_components
      # Need to add the calibration equation.


    b = torch.zeros((alphas.shape[-1]))
    
    mask = torch.ones(alphas_tmp.shape)
    index = []
    for i in range(alphas.shape[0]-1):
        mask[i*alphas.shape[1],:]=0
        mask[i*alphas.shape[1],i*alphas.shape[1]:(i+1)*alphas.shape[1]] = 1
        
        b[i*alphas.shape[1]] = 1
        index.append(i*alphas.shape[1])
    index = torch.tensor(index)
    
    for n_pixel, posteriors in tqdm(enumerate(outputs)):
        
        # posteriors (n_sources, n_classes)
        posteriors = torch.flatten(posteriors,0,1).unsqueeze(-1).repeat(1,1,betas.shape[-1]) # (n_sources*n_classes, n_composantes)
        A = (alphas_tmp - posteriors*betas_tmp).squeeze() 
      
        # A is n_equations, (n_models*n_classes), n_composantes
        # Need to add the calibration equation.
        A.index_fill_(0,index,1)
        A = A*mask
        
        #A = torch.cat((A.squeeze(), torch.ones((1,alphas.shape[-1]))), dim=0)
        #b = torch.cat((b,torch.ones((1))), dim=0)
        
        X = torch.tensor(scipy.linalg.solve(A, b))
        
        outputs_target.append(X)
        
           
    outputs_target = torch.stack(outputs_target)
    #outputs_target = torch.reshape(outputs_target, (n_classes, height, width))
    posteriors_target = torch.matmul(outputs_target,weights.T) * priors # (n_pixels, n_classes)
    posteriors_target = torch.transpose(torch.transpose(torch.reshape(posteriors_target, (height, width, n_classes)),0,2),1,2)
    return posteriors_target