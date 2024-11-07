import SGT
import cv2

if __name__ == '__main__':
    dataloader = SGT.Dataloader(batch_size=1, size=512, shuffle=True, description=True)

    train_dataloader = dataloader.get_train_dataloader()
    # val_dataloader = dataloader.get_val_dataloader()
    # test_dataloader = dataloader.get_test_dataloader()

    # EPOCHS = 2
    # for epoch in range(0, EPOCHS):
    for image, captions in train_dataloader: 
        image = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
        
        cv2.imshow('Image', image)

        # plt.imshow(image)
        # plt.title(caption)
    
        print(captions)
        
        if cv2.waitKey(0) == ord('q'):
            break