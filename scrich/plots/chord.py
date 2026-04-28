from d3blocks import D3Blocks
import pandas as pd
from IPython.display import display, HTML

# from playwright.sync_api import sync_playwright
# from pathlib import Path

def chord_diagram(adata, p, eps=0.01):

    states, mat = adata.uns['ccc_mat'][p]['states'], adata.uns['ccc_mat'][p]['mat']

    # transform CCC mat to pandas dataframe
    ccc_dict = {'source': [], 'target': [], 'weight': []}
    for i in range(len(states)):
        for j in range(len(states)):
            if mat[i][j] > eps:
                ccc_dict['source'].append(states[i])
                ccc_dict['target'].append(states[j])
                ccc_dict['weight'].append(mat[i][j])
    ccc_df = pd.DataFrame.from_dict(ccc_dict)
    # print(ccc_df)

    d3 = D3Blocks()
    d3.chord(ccc_df)
    d3.show()
    # html = d3.show(filepath=None, notebook=True)
    # display(HTML(html))

    # html_path = Path("d3blocks.html").resolve()
    #
    # with sync_playwright() as p:
    #     browser = p.chromium.launch()
    #     page = browser.new_page()
    #     page.goto(html_path.as_uri())
    #     page.wait_for_load_state("networkidle")  # wait for JS
    #     page.pdf(path="output.pdf", format="A4", print_background=True)
    #     browser.close()