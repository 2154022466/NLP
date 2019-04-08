# -*- coding:utf-8 -*-
__author__ = 'XF'

# 文本预处理

import jieba.posseg
import json
import os.path
import time
import re
import math

class Preprocess(object):

    def __init__(self):
        self.IOErrorNum = 0
        self.allErrorNum = 0
        self.OtherErrorNum = 0
        self.JiebaErrorNum = 0
        self.processNum = 0
        self.all_process_files = list()
        self.corpusPath = ''
        self.savePath = ''
        self.stopWordsPath = ''
        self.IDFPath = ''
        self.wordList = list()
        self.stopWords = list()
        self.exeTime = 0
        self.corpusWords = {}

    def init(self):
        with open("PreprocessLog.txt", "a", encoding="utf-8") as log:
            log.writelines("----------------------------------------------------------------------------------------\n")
        self.exeTime = time.time()

    def fenci(self, filePath):
        try:
            file = open(filePath, 'r')
            corpus = file.read()
            # 去除原始文本中的空白字符
            corpus = re.sub(r"\s", "", corpus)
        except IOError:
            corpus = ''
            self.IOErrorNum = self.IOErrorNum + 1
            self.allErrorNum = self.allErrorNum + 1
            self.writeLog_IOError(filePath, 'read')
        finally:
            file.close()
        if corpus:
            try:
                self.wordList = jieba.posseg.lcut(corpus)
                # print("Total words:", len(self.wordList))
                del corpus
            except Exception:
                self.allErrorNum = self.allErrorNum + 1
                self.JiebaErrorNum = self.JiebaErrorNum + 1
                self.writeLog_JiebaError("Jieba fenci error!")
            if self.wordList:
                self.delNoUseWords() # 根据词性删除特征项
                i = 0
                wordList = list(self.wordList)
                for pair in wordList:
                    self.wordList[i] = pair.word
                    i = i + 1
                del wordList
                self.delOnlyOneWords()
                if self.delStopWords():
                    self.textRepresentation()
                    splitPath = filePath.split("\\")
                    fileName = splitPath[len(splitPath) - 1]
                    if self.saveData(fileName):
                        return True
        return False

# 此处目前只是用停用词表在预处理阶段来减低文档向量的维度，还可以用其它降维的逻辑（利用词性降维）
    def delStopWords(self):
        if self.stopWords:
            # ---------------------------------------------
            wordList = list(self.wordList)
            # ---------------------------------------------
            startLen = len(self.wordList)
            for item in wordList:
                if item in self.stopWords:
                    self.wordList.remove(item)
            wordList.clear()
            endLen = len(self.wordList)
            # print("delStopWords:", startLen - endLen)
            return True
        return False

    # 独立字的过滤
    def delOnlyOneWords(self):
        wordList = list(self.wordList)
        startLen = len(self.wordList)
        for item in wordList:
            if len(item) < 2:
                self.wordList.remove(item)
        wordList.clear()
        endLen = len(self.wordList)
        # print("delOnlyOneWords:", startLen - endLen)

    # 根据词性进行过滤(连词，代词，介词，虚词，助词……)
    def delNoUseWords(self):
        ciX = ["y", "z", "w", "u", "t", "r", "q", "p", "o", "m", "e", "c"]
        wordList = list(self.wordList)
        startLen = len(self.wordList)
        for pair in wordList:
            if pair.flag in ciX:
                self.wordList.remove(pair)
        wordList.clear()
        endLen = len(self.wordList)
        # print("delNoUseWords:", startLen - endLen)

    def saveData(self, fileName):
        signal = False
        if self.wordList:
            curPath = os.path.join(self.savePath,fileName)
            try:
                file = open(curPath, 'w')
                file.writelines(self.wordList)
                signal = True
            except IOError:
                self.IOErrorNum = self.IOErrorNum + 1
                self.allErrorNum = self.allErrorNum + 1
                self.writeLog_IOError(curPath, 'write')
            finally:
                self.wordList.clear()
                file.close()
        return signal

    def recurGetFilesName(self, path):
        files = os.listdir(path)
        if files:
            for file in files:
                curPath = os.path.join(path, file)
                if os.path.isdir(curPath):
                    self.recurGetFilesName(curPath)
                else:
                    self.all_process_files.append(curPath)

    def textRepresentation(self):
        self.calculateTF()

    def calculateTF(self):
        i = 0
        N = len(self.wordList)
        if N:
            while i < len(self.wordList):
                word = self.wordList[i]
                num = self.wordList.count(word)
                self.wordList[i] = word + " " + str(num) + " " + "\n"

                # IDF
                value = self.corpusWords.setdefault(word, 0)  # 返回word的的值，不存在的话添加键并返回default
                self.corpusWords[word] = value + 1

                j = i + 1
                while j < len(self.wordList):
                    if self.wordList[j] == word:
                        self.wordList.remove(word)
                        continue
                    j = j + 1
                i = i + 1

    def saveIDF(self):
        idfList = list()
        for key in self.corpusWords:
            idfList.append(str(key + " " + str(self.corpusWords[key]) + "\n"))
        # self.corpusWords.clear()
        with open(self.IDFPath, "w") as idf:
            if idfList:
                idf.writelines(idfList)

    def calculateTF_IDF(self):
        N = len(self.corpusWords)
        self.all_process_files.clear()
        if os.path.isdir(self.savePath):
            self.recurGetFilesName(self.savePath)
        if self.all_process_files:
            for absPath in self.all_process_files:
                self.wordList.clear()
                with open(absPath, "r+") as tfFile:
                    self.wordList = tfFile.readlines()
                    tfFile.seek(0)
                    if self.wordList:
                        wordList = list()
                        for item in self.wordList:
                            item_list = item.split(" ")
                            word = item_list[0]
                            tf_raw = int(item_list[1].strip())
                            df = int(self.corpusWords[word])
                            if df:
                                tf_idf = (1 + math.log10(tf_raw)) * (math.log10(N/df))
                                wordList.append(item.strip("\n") + " " + str(tf_idf) + "\n")
                        tfFile.writelines(wordList)
                        wordList.clear()
                        self.wordList.clear()

    def characterSelection(self, DF = False, CHI = False, IG = False):
        """
        特征选择
        :param DF:文本频数
        :param CHI: 卡方检验
        :param IG: 信息增益
        :return:
        """
        if DF:
            self.DF()
        if CHI:
            self.CHI()
        if IG:
            self.IG()

    def DF(self):
        """
        基于假设：DF值低于最高阈值或高于最高阈值的特征性分别代表了两种极端情况：没有代表性，不具区分度，应从特征空间中删除
        :return:
        """
        MIN_THRESH = ""
        MAXTHRESH = ""


    def CHI(self):
        """
        通过度量特征项和类别之间的相关性，来判定一个特征项对于分类的重要程度
        基于假设：特征项和类别之间符合具有一阶自由度的卡方分布
        :return:
        """
        pass

    def IG(self):
        """
        通过考察类别Ci中出现与否特征项t,对整个分类系统的混乱程度的减低所起到的作用，IG越大，说明该特征对分类越有效
        :return:
        """
        pass


    def writeLog(self):
        with open("PreprocessLog.txt", 'a', encoding='utf-8') as log:
            log.writelines("    IOErrorNum:" + str(self.IOErrorNum) + "\n")
            log.writelines("    JiebaErrorNum:" + str(self.JiebaErrorNum) + "\n")
            log.writelines("    OtherErrorNum:" + str(self.OtherErrorNum) + "\n")
            log.writelines("    allErrorNum:" + str(self.allErrorNum) + "\n")
            log.writelines("    successfulNum:" + str(self.processNum) + "\n")
            log.writelines("    ExeTime:" + str(self.exeTime) + " s" + "\n")
            log.writelines("date:" + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + "\n")
            log.writelines("----------------------------------------------------------------------------------------\n")

    def writeLog_IOError(self, content, operation):
        with open("PreprocessLog.txt", 'a', encoding='utf-8') as log:
            log.writelines(operation + " Error:" + content + "\n")

    def writeLog_JiebaError(self, content):
        with open("PreprocessLog.txt", 'a', encoding='utf-8') as log:
            log.writelines("JiebaError:" + content + "\n")

    def writeOtherErrorLog(self, content):
        with open("PreprocessLog.txt", 'a', encoding='utf-8') as log:
            log.writelines("OtherError:" + content + "\n")

    def preProcess(self):
        try:
            self.init()
            with open('conf.json', 'r', encoding="utf-8") as config:
                config_str = json.load(config)
            self.corpusPath = config_str['corpusPath']
            self.savePath = config_str['savePath']
            self.stopWordsPath = config_str['stopWordsPath']
            self.IDFPath = config_str['IDFPath']
            with open(self.stopWordsPath, 'r', encoding="utf-8") as stopFile:
                self.stopWords = stopFile.readlines()
                i = 0
                for word in self.stopWords:
                    self.stopWords[i] = word.strip()
                    i = i + 1

            if os.path.isdir(self.corpusPath):
                 self.recurGetFilesName(self.corpusPath)

            # 读取corpus
            if self.all_process_files:
                for absPath in self.all_process_files:
                     if self.fenci(absPath):
                         self.processNum = self.processNum + 1

            # 写IDF文件
            self.saveIDF()

            # 计算TF-IDF
            self.calculateTF_IDF()
        except Exception as e:
            self.allErrorNum = self.allErrorNum + 1
            self.OtherErrorNum = self.OtherErrorNum + 1
            self.writeOtherErrorLog(str(e))
        finally:
            timeStamp = time.time()
            self.exeTime = timeStamp - self.exeTime
            self.writeLog()

if __name__ == "__main__":
    pp = Preprocess()
    pp.preProcess()
