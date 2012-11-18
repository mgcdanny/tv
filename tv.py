# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

import pandas as pd

# <codecell>

#import data files
Lbr = pd.read_table("/home/mgcdanny/Downloads/tv/liberalRegex.txt", sep='\t')
Cons = pd.read_table("/home/mgcdanny/Downloads/tv/ConsRegEx.txt", sep='\t')
#append data files to each other to make the main dataframe (df)
df = Lbr.append(Cons)
#creates an index (row numbers) from 0 through length of file
df = df.reset_index(drop=True)
#save memmory!
del(Lbr)
del(Cons)

# <codecell>

#aggregate the minutes viewd by county
#rows are counties, columns are channels, values are minutes
#if a county does not ever watch a show, a value of "NaN" is produced
xt = pd.crosstab(df.FIPS_CNTY_CD, df.CHNL_DESC, values=df.DRTN_MNTS, aggfunc=sum)

#create a list of channels that have viewership of more than 0 minutes
#sum each column of xt,  since NaN+anything = NaN and NaN is not greater than 0
#we are left with only the channels that have at least some viewership across all counties
chnl = []
for x in xt:
     if sum(xt[x]) > 0:
        chnl.append(x)

#Save memmory!
del(xt)

# <codecell>

#keeps only the common channels in the dataframe
df = df[df['CHNL_DESC'].isin(chnl)]

# <codecell>

#list of shows that need to be consolidated, 
#the first show of each tuple is the 'correct' name for which the other shows in that tuple will become
renames = [("SHOWTIME","SHOWTIME 3","SHOWTIME 2","SHOWTIMEOMEN","SHOWTIME NEXT","SHOWTIME BEYOND")
        ,("ESPN","ESPN NEWS","ESPNEWS","ESPN2","ESPNU","ESPN (73)")
        ,("ENCORE","ENCOREAM")
        ,("QVC", "QVC1","QVC2","QVC3")
        ,("DTV SPORTS","DTV SPORTS 702","DTV SPORTS 703")
        ,("STARZ","STRZ")
        ,("HBO","HBO 2", "HBO2")
        ,("DISNEY","DISNEYXD","THE DISNEY CHANNEL")
        ,("MTV","MTV2")
        ,("ENCORE","ENCOREERNS")
        ,("NATGEO","NATIONAL GEOGRAPHIC TV","NATIONAL GEOGRAPHIC CHANNEL","NAT GEOILD")
        ,("SPIKE","SPIKE TV")
        ,("FOOD","FOOD NETWORK")
        ,("ABC FAMILY","ABC FAMILY CHANNEL")
        ,("SYFY","SYFY CHANNEL")
        ,("TBS","TBS IN")
        ,("TMC", "TMC XTRA")
        ,("REELZ","REELZCHANNEL")
        ,("NFL","NFL NETWORK")
        ,("HALLMARK","HALLMARK CHANNEL")
        ,("COMEDY","COMEDY CENTRAL")
        ,("FOX NEWS CHANNEL","Fox News Channel")]

# <codecell>

#the loop that impliments the name change
for r in renames: 
    for n in r[1:]:
        temp = df[df['CHNL_DESC'] == n].index
        for t in temp:
            df.set_value(t, 'CHNL_DESC', r[0])

del(temp)

# <codecell>

#creates list of tuples with county code and state name
#never used this for anything but might good to have
fipName = zip(df.FIPS_CNTY_CD.unique(), ['MD','CA','NY','CO','MA','VA','GA','TN','TX','FL'])

# <codecell>

#create a new column called CNTY and just put a filler value 'hi' there
#will put the correct state name there
df['CNTY'] = 'hi'
#replace CNYT with correct state name
#might have been easier to create a seperate table and do an inner join
df.CNTY[df['FIPS_CNTY_CD'] == 24031]  = 'MD'
df.CNTY[df['FIPS_CNTY_CD'] == 6075]   = 'CA'
df.CNTY[df['FIPS_CNTY_CD'] == 36061]  = 'NY'
df.CNTY[df['FIPS_CNTY_CD'] == 8013]   = 'CO'
df.CNTY[df['FIPS_CNTY_CD'] == 25017]  = 'MA'
df.CNTY[df['FIPS_CNTY_CD'] == 51085]  = 'VA'
df.CNTY[df['FIPS_CNTY_CD'] == 13117]  = 'GA'
df.CNTY[df['FIPS_CNTY_CD'] == 47187]  = 'TN'
df.CNTY[df['FIPS_CNTY_CD'] == 48339]  = 'TX'
df.CNTY[df['FIPS_CNTY_CD'] == 12047]  = 'FL'

# <codecell>

#Normalize the data by rows:  divide each cell by total of respective row
#Returns the percent of total time a given row watched a given channel
f = lambda x: x/x.sum()
dfp = pd.pivot_table(data=df, values='DRTN_MNTS', rows=['CNTY','HH_ID','DVC_NMBR'], cols='CHNL_DESC', aggfunc=sum)
dfp = dfp.fillna(0)
dfp = dfp.apply(f,axis=1)
dfp = dfp.fillna(0)

# <codecell>

#alternate approach to creating matrix normalized matrix
#df2 = df.set_index(['FIPS_CNTY_CD','HH_ID','DVC_NMBR','CHNL_DESC'])
#df2 = df2.drop(['DT','CHNL_NMBR','STTM_LOCAL','ENDTM_LOCAL'],axis=1)
#df2 = df2.sum(level=['FIPS_CNTY_CD','HH_ID','DVC_NMBR','CHNL_DESC'])
#df2.unstack('CHNL_DESC').to_csv('testout.csv')
#df2 = df2.unstack('CHNL_DESC').apply(f, axis=1)
#df2 = df2.fillna(0)

# <codecell>

import pylab as pl
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA

# <codecell>

#take principal components of normalized data
pca = PCA(n_components=2, whiten=False)
X_r = pca.fit(dfp)
X_t = pca.fit(dfp).transform(dfp)

# <codecell>

#creates list of tuples with the state and scores on the first two princ comps: [(cnty, (load1, load2)), ... ] 
cnty = []
i = 0
for s in dfp.index:
    cnty.append((s[0], X_t[i]))
    i = i+1

# <codecell>

#create a list that will be used to make the json output file (for plotting with d3.js)
jl = []

for c in cnty:
     jd = {}
     jd['state'] = c[0]
     jd['sload1'] = c[1][0]
     jd['sload2'] = c[1][1]
     jd['chan'] = ""
     jd['load1']=""
     jd['load2']=""   
     #print jd
     jl.append(jd)

# <codecell>

import ujson as json

# <codecell>

#creates a DataFrame of the channels and their weights on the first two prin comps
dfc = pd.DataFrame(X_r.components_, columns=dfp.columns)
#only keeps the channels that have a total weights of greater than .05 
#many channels have weights of effectively zero, so not interested in keeping those in the results
g = lambda x: sum(abs(x))
dfc = dfc[dfc.T[(dfc.apply(g) > .05)].index]

d = [
         {"chan":i[0]  ,"load1":i[1][0], "load2":i[1][1], "sload1":"","sload2":"", "state":""}
         for i in dfc.to_dict().items()
    ]

# <codecell>

#create one json file with the channel weights and the home scores
d.extend(jl)
comps = json.dumps(d)

#actually creates the json file
fj = open('/home/mgcdanny/myCode/d3/book/mikedewar-getting_started_with_d3-9937443/visualisations/data/pca.json', 'w')
fj.write(comps)
fj.close()

# <codecell>

#make a quick plot to see what the data looks like visually
X_r = pca.fit(dfp).transform(dfp)
pl.figure
pl.scatter(X_r[:,0], X_r[:,1])
pl.legend()
pl.title('PCA of TV')
pl.show()

# <codecell>

from sklearn.ensemble import RandomForestClassifier

# <codecell>

#train a Random Forest on the normalized data
target = [t[0] for t in dfp.index[:]]
rf = RandomForestClassifier(n_estimators=20)
rf.fit(dfp, target)

# <codecell>

#predict the probabilities on the training set ("resubstitution rate") 
pred = rf.predict_proba(dfp)
#put the predictions in a DataFrame and add state names to columns and rows
rf_pred = pd.DataFrame(data=pred, columns=pd.unique(target), index=target)
#export to csv (or database)
rf_pred.to_csv('rf_pred.csv')

