main_dir = '/tf/notebooks/'
outdata_path = main_dir + 'final-project/LDA/data/gender/'

with open(os.path.join(outdata_path, 'data_preprocessed'), 'wb') as fp:
    pickle.dump(data_preprocessed, fp)
# with open(os.path.join(outdata_path, 'data_preprocessed'), 'rb') as fp:
#     data_preprocessed = pickle.load(fp)

word_index.save_as_text(outdata_path + 'dictionary')
# word_index = Dictionary.load_from_text(outdata_path+'dictionary')

optimal_model.save(outdata_path + 'lda_model_' + str(num_topics))
# optimal_model = LdaMulticore.load(outdata_path+'lda_model_'+str(num_topics))

all_speeches_topics_df.to_pickle(outdata_path + 'speeches_topics_' + str(num_topics))
# all_speeches_topics_df = pd.read_pickle(outdata_path+'speeches_topics_'+str(num_topics))

topics_df.to_pickle(outdata_path + 'topics_summary_' + str(num_topics))
# topics_df = pd.read_pickle(outdata_path+'topics_summary_'+str(num_topics))
