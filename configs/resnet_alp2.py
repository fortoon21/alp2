
def resnet_alp2_config(args):

    args.add_argument('--resnet_model', type=str, default='resnet18')
    args.add_argument('--img_size', type=int, default=96)

    # v_caption_patch_hangul
    args.add_argument('--alp_num', type=int, default=26)
    args.add_argument('--sb_num',  type=int, default=2)

    args.add_argument('--use_pretrained', type=bool, default=True)

    args.add_argument('--lr', type=float, default=1e-2)
    args.add_argument('--momentum', type=float, default=0.9)
    args.add_argument('--weight_decay', type=float, default=1e-4)
    args.add_argument('--accum_grad', type=int, default=1)

    opt = args.parse_args()

    return opt