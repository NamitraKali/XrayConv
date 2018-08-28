import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import copy
style.use('ggplot')

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class trainer():
    def __init__(self, model, loss=None, optimizer=None):
        self.model = model.to(device)
        self.loss_fn = loss.to(device)
        self.optimizer = optimizer
        self.reset_params = model.state_dict()
        self.snapshots = []
    
    def new_loss_fn(self, loss):
        self.loss_fn = loss.cuda()
    
    def new_optim(self, optimizer):
        self.optimizer = optimizer
    
    def get_snapshots(self):
        return self.snapshots
        
    def SGDR(self, data, cycles=1, cycle_len=1, cycle_mult=1,
             min_lr=1e-5, valid=None, snapshot=False):  
        
        loss_list = []
        lr_list = []
        iter_list = []
        iters = 0
        epoch = 0
        total_steps = len(data)
        total_epochs = sum([cycle_len*cycle_mult**i
                            for i in range(cycles)])
        
        for cycle in range(cycles):
            cur_epochs = cycle_len*(cycle_mult**cycle)
            
            # Add in Cosine Annealing to decrease the Learning Rate
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, cur_epochs*total_steps, eta_min=min_lr)

            if cycle > 0:
                self.model.load_state_dict(torch.load('train_model.ckpt'))

            for epochs in range(cur_epochs):
                print('#'*20 + f' EPOCH {epoch+1} ' + '#'*20)
                for i, (images, labels) in enumerate(data):
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                    # Change the Learning Rate
                    scheduler.step()
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Print progress
                    if (i + 1) % 20 == 0:
                        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                              .format(epoch + 1, total_epochs,
                                      i + 1, total_steps, loss.item()))

                    # Save the training version of the model
                    torch.save(self.model.state_dict(), 'train_model.ckpt')

                    # Add to our lists
                    iters += 1
                    iter_list.append(iters)
                    loss_list.append(float(loss.item()))
                    for params in self.optimizer.param_groups:
                        lr_list.append(params['lr'])
                        
                epoch += 1
                
                if valid is not None:
                    self.test(valid, valid=True)

            if snapshot == True:
                self.snapshots.append(copy.deepcopy(self.model.state_dict()))

        plt.plot(iter_list, loss_list)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title('Loss vs Iters')
        plt.show()

        plt.plot(iter_list, lr_list)
        plt.xlabel('iterations')
        plt.ylabel('lr')
        plt.title('Learning Rate vs Iters')
        plt.show()

    def lr_find(self, data, min_lr=1e-5, max_lr=1):
        loss_list = []
        lr_list = []
        iter_list = []
        total_steps = len(data)
        step_size = (1) / total_steps
        
        optimizer = optim.Adam(self.model.parameters())
        
        x_val = 0

        # Set our learning rate to the min_lr
        for params in optimizer.param_groups:
            params['lr'] = ((np.exp(10*x_val) - 1) / (np.exp(10) - 1)) + min_lr
        
        iters = 0
        for i, (images, labels) in enumerate(data):
            images = images.to(device)
            labels = labels.to(device)

            # Calculate the loss
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                     .format(1, 1, i + 1, total_steps, loss.item()))

            # Add to our lists
            iters += 1
            iter_list.append(iters)
            loss_list.append(float(loss.item()))
            for params in optimizer.param_groups:
                lr_list.append(params['lr'])
                params['lr'] = ((np.exp(10*x_val) - 1) / (np.exp(10) - 1)) + min_lr
            x_val += step_size
        
        fig, ax = plt.subplots()
        ax.semilogx(lr_list, loss_list)
        ax.grid()
        plt.xlabel('lr')
        plt.ylabel('loss')
        plt.title('Learning Rate vs Loss')
        plt.show()
        
        plt.plot(iter_list, lr_list)
        plt.xlabel('iterations')
        plt.ylabel('lr')
        plt.title('Learning Rate vs Iters')
        plt.show()
    
    def predict(self, model, data):
        model.to(device)
        with torch.no_grad():
            correct = 0
            total = 0
            loss = 0
            for data, labels in data:
                data = data.to(device)
                labels = labels.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                
                loss += F.nll_loss(output, labels).item()
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return loss
                
    
    def test(self, data, snapshot=None, valid=False):
        if snapshot is not None:
            snapshots = self.get_snapshots()[-snapshot:]
            
            net_list = []
            for weights in snapshots:
                net = copy.deepcopy(self.model)
                net.load_state_dict(weights)
                net.eval()
                net_list.append((net, self.predict(net, data)))
            
            # Sort snapshots to find the best performing ones
            net_list.sort(key=lambda tup: tup[1])
        
        else:
            self.model.eval()
        
        if snapshot:
            output_list = []
            
        with torch.no_grad():
            correct = 0
            total = 0
            loss = 0
            for data, labels in data:
                data = data.to(device)
                labels = labels.to(device)
                
                if snapshot is not None:
                    # Perform forward pass on each snapshot
                    output_list = [net[0](data).unsqueeze(0) for net in net_list]
                    #output_list = list(map(torch.exp, output_list))
                    
                    # Convert the predictions to labels
                    output = torch.mean(torch.cat(output_list), 0).squeeze()
                    predicted = output.data.max(1)[1]
                else:
                    output = self.model(data)
                    _, predicted = torch.max(output.data, 1)
                    
                loss += F.nll_loss(output, labels).item()
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            if valid:
                print('Validation Accuracy: {} %\nValidation Loss: {}'.format(100 * correct / total, loss))
            else:
                print('Test Accuracy: {} %\nTest Loss: {}'.format(100 * correct / total, loss))
