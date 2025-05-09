import torch
import argparse
import sys

from network import RnnFactory


class Setting:
    """ Defines all settings in a single place using a command line interface.
    """

    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])  # foursquare has different default args.

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        ###### settings ######
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim  # 10
        self.weight_decay = args.weight_decay  # 0.0
        self.learning_rate = args.lr  # 0.01
        self.epochs = args.epochs  # 100
        self.rnn_factory = RnnFactory(args.rnn)  # RNN:0, GRU:1, LSTM:2
        self.is_lstm = self.rnn_factory.is_lstm()  # True or False
        self.lambda_t = args.lambda_t  # 0.01
        self.lambda_s = args.lambda_s  # 100 or 1000


        self.transformer_nhid = args.transformer_nhid
        self.transformer_nlayers = args.transformer_nlayers
        self.transformer_nhead = args.transformer_nhead
        self.transformer_dropout = args.transformer_dropout
        self.attention_dropout_rate = args.attention_dropout_rate
        self.time_embed_dim = args.time_embed_dim
        self.user_embed_dim = args.user_embed_dim

        self.logit_adj_post= args.logit_adj_post
        self.tro_post_range = args.tro_post_range
        self.logit_adj_train = args.logit_adj_train
        self.tro_train= args.tro_train

        # self.poi_embedding_dim = args.poi_embedding_dim

        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.friend_file = './data/{}'.format(args.friendship)
        self.loader_file = './data/{}'.format(args.dataloader)
        self.max_users = 0  # 0 = use all available users
        self.sequence_length = 20  # 将用户的所有check-in轨迹划分成固定长度为20的多个子轨迹
        self.batch_size = args.batch_size
        self.min_checkins = 101

        # evaluation        
        self.validate_epoch = args.validate_epoch  # 每5轮验证一次
        self.report_user = args.report_user  # -1

        # log
        self.log_file = args.log_file
        self.model_file = args.model_file

        self.trans_loc_file = args.trans_loc_file  # 时间POI graph
        self.trans_loc_spatial_file = args.trans_loc_spatial_file  # 空间POI graph
        self.trans_user_file = args.trans_user_file
        self.trans_interact_file = args.trans_interact_file

        self.lambda_user = args.lambda_user
        self.lambda_loc = args.lambda_loc

        self.use_weight = args.use_weight
        self.use_graph_user = args.use_graph_user
        self.use_spatial_graph = args.use_spatial_graph

        ### CUDA Setup ###
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    def parse_arguments(self, parser):
        # training
        parser.add_argument('--gpu', default=0, type=int, help='the gpu to use')  # -1
        parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')  # 10
        parser.add_argument('--weight_decay', default=0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # 0.01
        parser.add_argument('--epochs', default=100, type=int, help='amount of epochs')  # 100
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')


        parser.add_argument('--transformer_nhid',
                        type=int,
                        default=32,
                        help='Hid dim in TransformerEncoder')
        parser.add_argument('--transformer_nlayers',
                        type=int,
                        default=2,
                        help='Num of TransformerEncoderLayer')
        parser.add_argument('--transformer_nhead',
                        type=int,
                        default=2,
                        help='Num of heads in multiheadattention')
        parser.add_argument('--transformer_dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for transformer')
        parser.add_argument('--attention_dropout_rate',
                        type=float,
                        default=0.1,
                        help='Dropout rate for attention_dropout_rate')
        parser.add_argument('--time_embed_dim',
                        type=int,
                        default=32,
                        help='Time embedding dimensions')
        parser.add_argument('--user_embed_dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')
        parser.add_argument('--logit_adj_post', help='adjust logits post hoc', type=int, default=1, choices=[0, 1])
        parser.add_argument('--tro_post_range', help='check diffrent val of tro in post hoc', type=list,
                        default=[0.25, 0.5, 0.75, 1, 1.5, 2])
        parser.add_argument('--logit_adj_train', help='adjust logits in trainingc', type=int, default=1, choices=[0, 1])
        parser.add_argument('--tro_train', default=1.0, type=float, help='tro for logit adj train')
        # parser.add_argument("--poi_embedding_dim", type=int, default=128)

        # data management
        parser.add_argument('--dataset', default='checkins-gowalla.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--dataloader', default='poi_loader-4sq-freq.pkl', type=str,
                            help='the dataloader under ./data/<dataloader.pkl> to load')
        parser.add_argument('--friendship', default='gowalla_friend.txt', type=str,
                            help='the friendship file under ../data/<edges.txt> to load')
        # evaluation        
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

        # log
        parser.add_argument('--log_file', default='./results/log_gowalla', type=str,
                            help='存储结果日志')
        parser.add_argument('--model_file', default='./model_log/model_gowalla', type=str,
                            help='存储模型日志')
        parser.add_argument('--trans_loc_file', default='./KGE/POI_graph/gowalla_scheme2_transe_loc_temporal_100.pkl', type=str,
                            help='使用transe方法构造的时间POI转换图')
        parser.add_argument('--trans_user_file', default='', type=str,
                            help='使用transe方法构造的user转换图')
        parser.add_argument('--trans_loc_spatial_file', default='', type=str,
                            help='使用transe方法构造的空间POI转换图')
        parser.add_argument('--trans_interact_file', default='./KGE/POI_graph/gowalla_scheme2_transe_user-loc_100.pkl', type=str,
                            help='使用transe方法构造的用户-POI交互图')
        parser.add_argument('--use_weight', default=False, type=bool, help='应用于GCN的AXW中是否使用W')
        parser.add_argument('--use_graph_user', default=False, type=bool, help='是否使用user graph')
        parser.add_argument('--use_spatial_graph', default=False, type=bool, help='是否使用空间POI graph')

    def parse_gowalla(self, parser):
        # defaults for gowalla dataset
        parser.add_argument('--batch-size', default=200, type=int,  # 200
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def parse_foursquare(self, parser):
        # defaults for foursquare dataset
        parser.add_argument('--batch-size', default=256, type=int,
                            help='amount of users to process in one pass (batching)')  # 1024
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')
        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for transition graph')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user graph')

    def __str__(self):
        return (
                   'parse with foursquare default settings' if self.guess_foursquare else 'parse with gowalla default settings') + '\n' \
               + 'use device: {}'.format(self.device)
