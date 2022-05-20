import os
import matplotlib.pyplot as plt
import seaborn as sns

def createDir(path):
    '''
    Tries to create a directory (relative to root)    
    '''
    try:
        os.mkdir(path)
    except FileExistsError:
        print(path, 'directory already exists')
        
        
def display_model_trainTestGraphs(results):
    '''
    Displays train vs validation graphs for accuracy and loss
    '''
    train_loss = results.history['loss']
    train_acc = results.history['accuracy']    
#     train_prec = results.history['precision']
#     train_recall = results.history['recall']
    
    val_loss = results.history['val_loss']
    val_acc = results.history['val_accuracy']
#     val_prec = results.history['val_precision']
#     val_recall = results.history['val_recall']

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    sns.lineplot(x=results.epoch, y=train_loss, ax=ax1, label='train_loss')
    sns.lineplot(x=results.epoch, y=train_acc, ax=ax2, label='train_accuracy')
#     sns.lineplot(x=results.epoch, y=train_prec, ax=ax3, label='train_precision')
#     sns.lineplot(x=results.epoch, y=train_recall, ax=ax4, label='train_recall')

    sns.lineplot(x=results.epoch, y=val_loss, ax=ax1, label='val_loss')
    sns.lineplot(x=results.epoch, y=val_acc, ax=ax2, label='val_accuracy')
#     sns.lineplot(x=results.epoch, y=val_prec, ax=ax3, label='val_precision')
#     sns.lineplot(x=results.epoch, y=val_recall, ax=ax4, label='val_recall')
    
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
#     ax3.set_title('Precision')
#     ax4.set_title('Recall')
    ax1.legend();