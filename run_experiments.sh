# # baseline: 
# python sam_optim.py --max_sam_masks=768 --M=512 --w_fit=1 --temp=1 --lr=10 --steps=1000 --w_reg=0.1 --outdir=runs/run0
# # reduce num target masks:
# python sam_optim.py --max_sam_masks=512 --M=512 --w_fit=1 --temp=1 --lr=10 --steps=1000 --w_reg=0.1 --outdir=runs/run1
# don't use softmax: 
python sam_optim_tree.py --sam_masks=512 --M=512 --w_fit=1 --temp=1 --lr=10 --steps=1000 --w_reg=0.1 --outdir=runs/run2
# add tree loss: 
python sam_optim_tree.py --sam_masks=512 --M=512 --w_fit=1 --temp=1 --lr=10 --steps=1000 --w_reg=0.1 --w_tree=0.1 --outdir=runs/run3