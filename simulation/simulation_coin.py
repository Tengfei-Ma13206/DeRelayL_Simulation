import math
from itertools import groupby
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import itertools

# 设置随机数种子
seed = 42
random.seed(seed)
np.random.seed(seed)

class Model():
    def __init__(self, version: int, baseModel, delta_performance_number: float, MO, T) -> None:
        self.baseModel = baseModel
        self.version = version
        self.MO = MO 
        self.T = T
        self.rank = -1
        if baseModel == None:
            self.performance_number = delta_performance_number
        else:
            self.performance_number = delta_performance_number + baseModel.performance_number

    def setRank(self, rank):
        self.rank = rank

    def get_performance(self):
        return math.exp(self.performance_number) / (1 + math.exp(self.performance_number))
    
    @staticmethod
    def delta_performance_number():
        return np.random.normal(loc=1, scale=1)

class Deposit():
    def __init__(self, MO, MO_deposit, T, T_deposit):
        self.MO = MO
        self.MO_deposit = MO_deposit
        self.T = T
        self.T_deposit = T_deposit


class DepositBlock():
    def __init__(self, Miner, previousBlock, depositList):
        self.Miner = Miner
        self.previousBlock = previousBlock
        self.depositList = depositList
        self.reward_per_deposit = 0.001

    def incentive(self):
        self.Miner.number_of_coins += len(depositList) * self.reward_per_deposit
        
class EncryptionBlock():
    def __init__(self, Miner, previousBlock, modelList):
        self.Miner = Miner
        self.previousBlock = previousBlock
        self.FHEpk = 'FHEpk'
        self.modelHashList = modelList # model list
        self.reward_per_hash = 0.001

    def incentive(self):
        self.Miner.number_of_coins += len(self.modelHashList) * self.reward_per_hash

class TestingBlock():
    def __init__(self, Miner, previousBlock, modelList):
        self.Miner = Miner
        self.previousBlock = previousBlock
        self.hashFHEMlist = modelList
        self.reward_per_hash = 0.001

        self.testingCases = [(f'TI{i}', f'TO{i}') for i in range(100)]
        self.reward_per_case = 0.001

    def incentive(self):
        self.Miner.number_of_coins += len(self.testingCases) * self.reward_per_case + len(self.hashFHEMlist) * self.reward_per_hash

class SettlementBlock():
    def __init__(self, Miner, previousBlock, performanceList, num_cases):
        self.Miner = Miner
        self.previousBlock = previousBlock
        self.performanceList = performanceList # already sorted
        self.reward_per_model = 0.001
        self.num_cases = num_cases
        self.reward_per_test = 0.001
    
    def incentive(self):
        self.Miner.number_of_coins += len(self.performanceList) * self.reward_per_model + len(self.performanceList) * self.num_cases * self.reward_per_test

    def sortedPerformanceList(self):
        pass

class BlockChain():
    def __init__(self) -> None:
        self.blockchain = []

    def addBlock(self, block):
        self.blockchain.append(block)

    def getLatestBlock(self):
        if len(self.blockchain) == 0:
            return None
        else:
            return self.blockchain[len(self.blockchain)-1]

class Pariticipant():
    def __init__(self) -> None:
        self.number_of_coins = 0
        self.expectedUtility = 0

        self.bid_T = -1.0
        self.bid_MO = -1.0

        self.GainedModels = []
        self.EncryptedModels = []

        self.MO_budget = 0.001

        self.bestModelPerformance = 0

    def T_allin(self):
        self.bid_T = self.number_of_coins
        self.number_of_coins = 0

    def MO_NoneIn(self):
        self.bid_MO = 0.0

    def getDepositBack(self, identity: str):
        if identity == 'MO':
            self.number_of_coins += self.bid_MO

        if identity == 'T':
            self.number_of_coins += self.bid_T

    def T_bid_strategy(self, round):
        if len(self.GainedModels) == 0:
            latest_version = 0
        else:
            latest_version = self.GainedModels[-1].version 

        if len(self.GainedModels) == 0:
            V_now = 0
        else:
            V_now = latest_version

        if self.number_of_coins <= (round - V_now + 1 ): # b^T try the best to be greater than (V_RecM - V_now^T)
            self.bid_T = self.number_of_coins
        else:
            self.bid_T = (round - latest_version + 1 )

    def MO_bid_strategy(self, selection_num_limit):
        eps = 1e-12 # prevent negative number of coins
        if self.number_of_coins <= self.MO_budget:
            self.bid_MO = self.number_of_coins / (selection_num_limit + eps)
        else:
            self.bid_MO = self.MO_budget / selection_num_limit

    def calculate_bestModelPerformance(self):
        totalModelList = self.GainedModels + self.EncryptedModels
        if len(totalModelList) == 0:
            return 0.0
        self.bestModelPerformance = max(m.performance_number for m in totalModelList)
        return self.bestModelPerformance
        # self.bestModelPerformance = math.exp(self.bestModelPerformance) / (1 + math.exp(self.bestModelPerformance))
        # return self.bestModelPerformance
    def get_latest_version(self) -> int:
        totalModelList = self.GainedModels + self.EncryptedModels
        if len(totalModelList) == 0:
            return -1
        return max(m.version for m in totalModelList)


class Participants():
    def __init__(self, total_number_of_participants=1024) -> None:
        self.Participants = [Pariticipant() for _ in range(total_number_of_participants)]
        


def sort_bid_T(bidList: list):
    sorted_bidList = sorted(bidList, key=lambda x: x.bid_T, reverse=True)
    result = []
    for _, group in groupby(sorted_bidList, key=lambda x: x.bid_T):
        group_list = list(group)
        random.shuffle(group_list)
        result.extend(group_list)

    return result

def sort_performance_MO(performanceList: list):
    sorted_performanceList = sorted(performanceList, key=lambda x: x.GainedModels[-1].performance_number, reverse=True)
    return sorted_performanceList


def plot_participants_data(data, save_directory='pic/'):
    rounds = len(data)
    num_participants = len(data[0][0])

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Extract data for each participant
    coins_data = [d[0] for d in data]
    version_data = [d[1] for d in data]

    # Set global plot settings
    plt.rc('font', family="Times New Roman", size=55)  # 55 for coin, 42 for version

    # Plotting the coins per participant over rounds (Line Chart)
    plt.figure(figsize=(31.2, 16))
    for participant_index in range(num_participants):
        plt.plot(range(1, rounds + 1), [coins_data[round_idx][participant_index][0] for round_idx in range(rounds)], alpha=0.6)
    plt.title('Coins per Participant Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Coins')
    plt.grid(True)

    # Set scientific notation for y-axis
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    # 设置 y 轴的刻度为 5e5
    ax = plt.gca()  # 获取当前的轴对象
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5e5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    plt.xlim(1, rounds)  # Ensure the x-axis limits match the available data rounds

    plt.tight_layout()
    ax.set_ylim(0, 4e6)
    coins_plot_path = os.path.join(save_directory, 'coins_per_participant_over_rounds.pdf')
    plt.savefig(coins_plot_path, dpi=300, bbox_inches="tight", format='pdf')
    plt.close()

    # # Plotting the version distribution per participant over rounds (Stacked Bar Chart)
    # plt.figure(figsize=(31.2, 16))  # Increase width to provide more horizontal space

    # # Find the latest version per round for plotting
    # latest_version_per_round = [
    #     max([max([version for version in participant_versions if version >= 0], default=-1) for participant_versions in round_versions])
    #     for round_versions in version_data
    # ]

    # for round_idx in range(rounds):
    #     version_counts = [0] * 13  # 11 bins for latest version to latest version -10, one for older, and one for None
    #     for participant_versions in version_data[round_idx]:
    #         latest_version = latest_version_per_round[round_idx]
    #         version_diff = latest_version - participant_versions[0]

    #         # Cap the version difference to fall within the range 0 to 11 (where 11 means "Older" and 12 means "None")
    #         if participant_versions[0] == -1:
    #             version_diff = 12  # Special case for participants with no model (None)
    #         elif version_diff < 0 or version_diff > 10:
    #             version_diff = 11  # Treat versions older than latest version -10 as "Older"

    #         version_counts[version_diff] += 1

    #     # Convert counts to percentages
    #     version_percentages = [count / num_participants * 100 for count in version_counts]
    #     bar_labels = ['Latest', 'Latest-1', 'Latest-2', 'Latest-3', 'Latest-4', 'Latest-5',
    #                   'Latest-6', 'Latest-7', 'Latest-8', 'Latest-9', 'Latest-10', 'Older', 'None']

    #     # Plot the stacked bar for this round
    #     bottom = np.zeros(len(version_percentages))
    #     for i in range(len(version_percentages)):
    #         if i == 12:  # Special case for None (white color)
    #             color_value = 'white'
    #         elif i == 11:  # Older versions
    #             color_value = 'lightgray'
    #         else:  # Latest versions with gradient colors
    #             color_value = plt.cm.Blues(255 - i * 20)
    #         plt.bar(round_idx + 1, version_percentages[i], bottom=bottom, color=color_value,
    #                 label=bar_labels[i] if round_idx == 0 else "", width=1.8)  # Increase width to 1.8 for wider columns
    #         bottom += version_percentages[i]

    # plt.title('Model Version Distribution Over Rounds')
    # plt.xlabel('Round')
    # plt.ylabel('Percentage of Participants (%)')
    # plt.grid(True, axis='y')
    # plt.yticks(np.arange(0, 101, 10))  # Set y-axis ticks to show percentages at intervals of 10
    # plt.xticks(np.arange(0, rounds + 1, 10))  # Set x-axis ticks to show at intervals of 10 rounds

    # # Limit x-axis to show only the relevant rounds without extending
    # plt.xlim(1, rounds)

    # # Adjust legend size and position for better readability
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.03, 0.5), loc='center left', fontsize=34)  # Larger font for legend and adjusted position

    # plt.tight_layout(pad=5.0)  # Adjust padding to ensure enough space for the chart and legend
    # version_plot_path = os.path.join(save_directory, 'version_distribution_per_participant_over_rounds.pdf')
    # plt.savefig(version_plot_path, dpi=300, bbox_inches="tight", format='pdf')
    # plt.close()

    # return coins_plot_path, version_plot_path



if __name__ == '__main__':
    blockchain = BlockChain()
    s = 0.5
    num_participants = 256
    participants = Participants(num_participants)
    GenesisBlockOwner = participants.Participants[0]
    
    selection_num_limit = 4
    num_MO_T_limit = num_participants // 2

    num_simulated_round = 5000 # 100 for version , 5000 for coins
    print(f"There are {num_simulated_round} rounds to simulate.")
    MO_list = [GenesisBlockOwner] # initial MO list
    Models = [Model(0, None, 0.0, GenesisBlockOwner, GenesisBlockOwner)] # initial model
    GenesisBlockOwner.GainedModels.append(Models[0]) # Genesis Block Owner have initial model

    data = []


    for round in range(num_simulated_round):
        print(f'-------- round {round} start ----------')

        # 1. Role assignment: MO T Miner
        # print('0. Basic info:')
        print(f'There are {len(MO_list)} MO.')
        participants_except_MO = [p for p in participants.Participants if p not in MO_list]
        # print(f'There are {len(participants_except_MO)} participants except MO.')
        MO_T_list = MO_list.copy()
        T_list = random.sample(participants_except_MO, num_MO_T_limit-len(MO_list))
        print(f'There are {len(T_list)} Trainers.')
        MO_T_list.extend(T_list)
        Miner_list = [p for p in participants.Participants if p not in MO_T_list]
        # print(f'There are {len(Miner_list)} Miners.')

        # 2. Auction
        # 2.1 bidding
        print('1. Bidding:')
        for t in T_list:
            t.T_bid_strategy(round)
            # print(f'T has bid = {t.bid_T} and number_of_coins = {t.number_of_coins}')
        for mo in MO_list:
            mo.MO_bid_strategy(selection_num_limit)
            # print(f'MO has bid = {mo.bid_MO} and number_of_coins = {mo.number_of_coins}')
        # 2.2 Matching
        bid_T = sort_bid_T(T_list) # T list sorted by bid
        performance_MO = sort_performance_MO(MO_list) # MO list sorted by latest model performance
        MO_T_cooperation = {mo: [] for mo in performance_MO}

        bid_T_left = bid_T.copy()
        for mo in performance_MO:
            for _ in range(selection_num_limit):
                if bid_T_left:
                    selected_T = bid_T_left.pop(0)
                    MO_T_cooperation[mo].append(selected_T)
            # print(f'MO matches with {len(MO_T_cooperation[mo])} Trainers')
        # 2.3 Consume coines
        for mo in performance_MO:
            mo.number_of_coins -= len(MO_T_cooperation[mo]) * mo.bid_MO
            for t in MO_T_cooperation[mo]:
                t.number_of_coins -= t.bid_T



        # 3. Deposit
        depositList = []
        for mo in performance_MO:
            for t in MO_T_cooperation[mo]:
                depositList.append(Deposit(mo, mo.bid_MO, t, t.bid_T))
        
        # print(f'There are {len(depositList)} pieces of deposits.')

        DBM = random.sample(Miner_list, 1)[0]
        # print(f'DBM has {DBM.number_of_coins} coins before incentive.')
        DB = DepositBlock(DBM, blockchain.getLatestBlock(), depositList)
        blockchain.addBlock(DB)
        DB.incentive()
        # print(f'DBM has {DBM.number_of_coins} coins after incentive.')
        # print(f'There are {len(blockchain.blockchain)} blocks. And Deposit Block is added.')


        

        # 4. Training
        trained_rate = 0.9
        TrainedModelList = []
        for mo in performance_MO:
            for t in MO_T_cooperation[mo]:
                if trained_rate > random.uniform(0, 1):
                    TrainedModel = Model(round+1, mo.GainedModels[-1], Model.delta_performance_number(), mo, t)
                    # print(f'Base model has performance = {TrainedModel.baseModel.performance_number}.')
                    # print(f'Trained model has performance = {TrainedModel.performance_number}.')
                    TrainedModelList.append(TrainedModel)
                    t.GainedModels.append(TrainedModel)
        
        # print(f'There were {len(TrainedModelList)} trained models just now.')

        # 5. Encryption
        EBM = random.sample(Miner_list, 1)[0]
        # print(f'EBM has {EBM.number_of_coins} coins before incentive.')
        EB = EncryptionBlock(EBM, blockchain.getLatestBlock(), TrainedModelList)
        blockchain.addBlock(EB)
        EBM.EncryptedModels.extend(TrainedModelList)
        EB.incentive()
        # print(f'EBM has {EBM.number_of_coins} coins after incentive.')
        # print(f'There are {len(blockchain.blockchain)} blocks. And Encryption Block is added.')

        # 6. Testing
        TBM = random.sample(Miner_list, 1)[0]
        # print(f'TBM has {TBM.number_of_coins} coins before incentive.')
        TB = TestingBlock(TBM, blockchain.getLatestBlock(), TrainedModelList)
        blockchain.addBlock(TB)
        TB.incentive()
        # print(f'TBM has {TBM.number_of_coins} coins after incentive.')
        # print(f'There are {len(blockchain.blockchain)} blocks. And Testing Block is added.')

        # 7. Settlement
        SBM = random.sample(Miner_list, 1)[0]
        # print(f'SBM has {SBM.number_of_coins} coins before incentive.')
        performanceList = sorted(TrainedModelList, key=lambda x: x.performance_number, reverse=True)
        SB = SettlementBlock(SBM, blockchain.getLatestBlock(), performanceList, len(TB.testingCases))
        blockchain.addBlock(SB)
        SB.incentive()
        # print(f'SBM has {SBM.number_of_coins} coins after incentive.')
        # print(f'There are {len(blockchain.blockchain)} blocks. And Settlement Block is added.')

        MO_list = []
        for i in range(math.floor(len(TrainedModelList) * s)):
            MO_list.append(performanceList[i].T)
            # print(f'T has {performanceList[i].T.number_of_coins} coins before getting deposit back.')
            performanceList[i].T.getDepositBack('T')
            # print(f'T has {performanceList[i].T.number_of_coins} coins after getting deposit back.')

            # print(f'MO has {performanceList[i].MO.number_of_coins} coins before getting deposit back.')
            performanceList[i].MO.getDepositBack('MO')
            # MO incentive
            performanceList[i].MO.number_of_coins += 1
            model_selected = performanceList[i]
            while model_selected.baseModel != None:
                model_selected = model_selected.baseModel
                model_selected.MO.number_of_coins += 1
                # if model_selected.MO == GenesisBlockOwner:
                #     print(f"is Genesis")
            # print(f'MO has {performanceList[i].MO.number_of_coins} coins after getting deposit back.')

            # print(f"Genesis Block Owner has {GenesisBlockOwner.number_of_coins} coins.")

        coins_per_participant = [[p.number_of_coins] for p in participants.Participants]
        version_per_participant = [[p.get_latest_version()] for p in participants.Participants]
        data.append([coins_per_participant, version_per_participant])
    plot_participants_data(data)



# 给之前所有的 MO都要发币，不要只给 current MO 发币
# 不能随便发币，否则所有人都可以不劳而获模型
# b^T > V_RecM - V_now 出价策略， taker 策略
# fig1: 每个人过往 Utility sum 作为满意指数 per round per participant
# fig2: 每个人拥有的最佳模型性能  per round per participant









        

        




        

