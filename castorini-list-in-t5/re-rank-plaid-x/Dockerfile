#docker build -t list-in-t5-re-rank-plaid-x -f re-rank-plaid-x/Dockerfile .
FROM mam10eks/list-in-t5:0.0.1

RUN pip3 uninstall -y tira && pip3 install tira && pip3 install --no-deps ir-datasets

ADD re-rank-plaid-x/list-in-t5-rerank-plaid-x.sh re-rank-plaid-x/prepare-rerank-file-from-plaid-x.py /

ENTRYPOINT [ "/list-in-t5-rerank-plaid-x.sh", "$inputDataset", "$outputDir" ]
