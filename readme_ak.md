radial-t-unet repo: (only use 1 gpu)
	sample branch: to define different types of model change the TYPE of model in config file
		TYPE : Swin (trains swin)
		TYPE : Darswin_ra (Rad-Azimuth merge)
        TYPE : Darswin_az (Azimuth merge)

	pixel branch:

        Bug : in lazy tensor in dataloader CUDA error in Keops

        Calculate the precomputing grid for samples using the Calc_matrix.py in pixel branch


To change the index gpu : 
    I will rectify it but for now, just change the device for all tensors in Trainer.py, Train.py and (Swin_Transforer_az.py, Swin_Transforer_az.py, swin_transformer_unet_skip_expand_decoder_sys.py ) just do ctr+f and find "cuda:1" and change it to the desired gpu index. If you have only one gpu the index will be cuda:0