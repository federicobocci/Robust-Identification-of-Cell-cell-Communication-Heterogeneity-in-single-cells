'''
test the database import functions
'''
import tf_tar_functions as ttf
'''
objective: find the top N TFs for an input pathway
'''

def main():
    p = 'IL1'
    lig_dct, rec_dct = ttf.cellchat_DB(species='mouse')
    tf_df = ttf.layer_2_DB(species='mouse')

    print(lig_dct[p], rec_dct[p])

    tf_list = ttf.get_TF(rec_dct, lig_dct, tf_df, p)

    print(len(tf_list))

if __name__=='__main__':
    main()