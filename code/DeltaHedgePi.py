#necessary imports
import pandas_datareader as pdr
from pandas_datareader.data import DataReader as dr
import datetime as dt
from datetime import date
from yahoo_fin import stock_info as si
from yahoo_fin import options
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  
from AutoDiff import AutoDiff, logist, logN


#first we define Black Scholes function to calculate delta and vega pull uptodate financial data

def BS_Delta(ticker, exp_year, exp_month, exp_day, op_type, strike):
    """ Obtain upto date financial data for stock price, option price, volatility, risk free rate
    and calculates delta and vega from Black Scholes equation. Returns delta, vega, time to expiration,
    most uptodate share price and corresponding option price.
    
    INPUTS:
    ======
    ticker:     accepts both upper and lower case (string)
    exp_year:   year of expiration (int)
    exp_month:  month of expiration (int)
    exp_day:    day of expiration (int)
    opt_type:   'puts' for put and 'calls' for call(string)
    strike:     price at which the opion can be excercised (int)
    
    OUTPUT:
    ======
    delta_bs:   delta calculated using Black Scholes equation (partial derivative w.r.t. share price)
    vega:       vega calculated using Black Scholes euqation (partial derivative w.r.t. share price vol)       
    T_t:        time to option expiration as fraction of the year
    S:          share price
    C:          option price  
    
    NOTES:
    ====== 
    This fuction pulls most upto date financial data form yahoo finance. The option values are pulled 
    pulled during the market hours (9:30am-4pm, mon-fri).  When the market is closed, the function
    will ask the user to input the option price. Also, just like Black Sholes equation, this function
    relies on static volatlity measured as historical s.d. of the share price over the past year (i.e. we
    pull in a year share prices and calculate standard deviation of that for sigma).
    This function also requires the following pacakges for data retrieval in addition to pandas, numpy amd
    datetime:
    import sys
    !{sys.executable} -m pip install yahoo_fin
    !{sys.executable} -m pip install requests_html
    
    EXAMPLE:
    =======
    We are not writing a formal doc test here because the outputs are not static. This illustration should
    help the user understand how the pakage work. 
    
    Let's suppose you wanted to run a GE call expriing on Jan 17, 2020, with  strike 12.
    ticker='GE'
    exp_year=2020
    exp_month=1
    exp_day=17
    exp_date=str(str(exp_month)+'/'+str(exp_day)+'/'+str(exp_year))
    op_type='calls'
    strike=12
    
    
   
    delta_bs, vega, T_t, S, C = BS_Delta(ticker, exp_year, exp_month, exp_day, op_type, strike) 
    

 0.2803917681986156 0.009142607098052206 0.10684931506849316 11.100000381469727 0.4027907848942922
    
    !!since the function is pulling live data for this doctest the output will keep changing!!
    
    
    """
    
    
    #convert to exp_date format
    exp_date=str(str(exp_month)+'/'+str(exp_day)+'/'+str(exp_year))
    #get current risk free rate
    #we are using 10tyear treasury which is industry standard
    tbills = ['DGS10']
    yc = dr(tbills, 'fred') 
    risk_free=np.array(yc)[-1][0]/100
    #get current date
    today=date.today()
    #get stock price
    stock = yf.Ticker(ticker)
    S=stock.history(period='min')['Open'][0]
    #get price of option closest to the same date/strike as our option
    try:
        table=options.get_options_chain(ticker,exp_date )
        strikes=table[op_type]['Strike']
        closest_strike=min(strikes, key=lambda x: abs(x-strike))
        index=table[op_type][table[op_type]['Strike'].isin([closest_strike])==True].index[0]
        C=np.mean(table[op_type][['Bid','Ask']].iloc[index])
    except:
        print('Could not find live option price, will calculate implied price instead')
        C=0 #place holder value, we will calculate black scholes implied value below
        
        #print('could not find live price for this option type. please enter best available estimate:')
        #C=float(input(""))
    def vol_est(today, ticker):
    #get volatility of underlying assets
        end = today
        start=dt.datetime.now() - dt.timedelta(days=365)
        prices=list(pdr.get_data_yahoo(ticker, start, end)['Adj Close'])
        returns=[]
        i=0
        while i < (len(prices)-1):
            r=np.log(prices[i+1]/prices[i])
            returns.append(r)
            i=i+1
        vol=np.std(returns)*(250**(1/2))
        return vol
    #setting up all the inputs for the Black Scholes formula
    vol=vol_est(today, ticker)
    T=dt.date(exp_year, exp_month, exp_day,)
    t=today
    r=risk_free
    T_t=(T-t).days/365
    #first we will calculate d1 and d2
    d1=(np.log(S/strike)+(r+vol**2/2)*(T_t))/(vol*((T_t)**(1/2)))
    d2=d1-vol*(T_t)**(1/2)
    if C==0:
        if op_type == 'calls':
            C=S*(1/(2*np.pi)**(1/2))*np.e**(-d1)*d1 - strike*np.e**(-r*(T_t))*(1/(2*np.pi)**(1/2))*np.e**(-d1)*d1
        else:
            if op_type =='puts':
                C=strike*np.e**(-r*T_t)*(1/(2*np.pi)**(1/2))*np.e**(d2)*(-d2)-S*(1/(2*np.pi)**(1/2))*np.e**(d1)*(-d1)
    #now we will calculate delta depending on option type
    if op_type=='calls':
        delta_bs=1/((2*np.pi)**(1/2))*np.e**(-d1**2/2)
    else:
        delta_bs=1/((2*np.pi)**(1/2))*np.e**(-d1**2/2)-1
    #calculate vega (same formula for puts and calls)
    vega=S/100*np.e**(-T_t)*(T_t**(1/2))*delta_bs
                                                          
    return delta_bs, vega, T_t, S, C  

#now we build a volatility surface plot function

def Volatility_Surface(ticker, exp_year, exp_month, exp_day, op_type, strike, price):
    """ This function calculates and plots 2 volatility 3D surface plots of stock volatlity as calculated 
    by Bharadia and Corrado approximations vs. time and share price of the underlying asset. 
    
    INPUTS:
    ======
    ticker:     accepts both upper and lower case (string)
    exp_year:   year of expiration (int)
    exp_month:  month of expiration (int)
    exp_day:    day of expiration (int)
    opt_type:   'puts' for put and 'calls' for call(string)
    strike:     price at which the opion can be excercised (int)
    price:      option price (float)
    
    OUTPUT:
    ======
    3Dplot Bharadia:  surface plot for estimated stock price volatilty over time and share price space
    3Dplot Corrado:   surface plot for estimate of stock volatility over time and share price space.
    
    NOTES:
    ====== 
    This function pulls live stock price data and requires the following pacakges to run properly:
    pandas
    numpy
    mpl_toolkits
    matplotlib.pyplot 
    datetime
    yahoo_fin 
    requests_html 

    """
    
    
    #current data
    today=date.today()
    T=dt.date(exp_year, exp_month, exp_day,)
    #cacl time to expiration
    t=today
    T_t=(T-t).days/365
    #calculate volatility
    def vol_est(today, ticker):
            #get volatility of underlying assets
                end = today
                start=dt.datetime.now() - dt.timedelta(days=365)
                prices=list(pdr.get_data_yahoo(ticker, start, end)['Adj Close'])
                sd=np.std(prices)
                return sd
    vol=vol_est(today, ticker)
    stock = yf.Ticker(ticker)
    S=stock.history(period='min')['Open'][0]
    S_low=int(S-2*vol)
    S_high=int(S+2*vol)
    S_range=list(range(S_low, S_high))
    days_range=list(range(1,int((T-t).days)))
    C=price
    vol_s=[]
    vol_c=[]
    #get range of volatlities for both methods
    for i, t in enumerate(days_range):
            vol_t=[]
            vol_tc=[]
            for i, s in enumerate(S_range):
                    vol_simple=(2*np.pi/(t/365))**(1/2)*((C-(s-strike))/2)/(s-(s-strike)/2)
                    vol_t.append(vol_simple)
                    vol_complex=((2*np.pi/(t/365))**2)**(1/2)*1/(s+strike)*(((C-(s-strike)/2+((C-(s-strike)/2))**2-(s-strike)**2/np.pi)**2)**(1/2))**(1/2)
                    vol_tc.append(vol_complex)
            vol_s.append(vol_t)
            vol_c.append(vol_tc)
        
    data = np.array(vol_s)
            
            #prepare the data for plotting
            
            #plot Bharadia approximation
    x, y = np.meshgrid(S_range,days_range)
    fig = plt.figure(figsize=(12,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, data, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_title('Volatility Surface Simple (Bharadia et al. )')
    ax.set_xlabel('stock price')
    ax.set_ylabel('days to expiration')
    ax.set_zlabel('volatility')
    ax.view_init(30, 35)
    plt.show(fig)
            
            
    data = np.array(vol_c)
            
            #plot Corrado approximation
    x, y = np.meshgrid(S_range,days_range)
    fig = plt.figure(figsize=(12,10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, data, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_title('Volatility Surface Complex (Corrado et al. )')
    ax.set_xlabel('stock price')
    ax.set_ylabel('days to expiration')
    ax.set_zlabel('volatility')
    ax.view_init(30, 35)
    plt.show(fig)
            
def OptionsRun():
    """ This an interactive enclosing functionn for two nested functions that prompts the user 
    to enter the details about their option positive and returns 3 deltas calculated with Black 
    Scholes formula, Bharadia and Corrado estimation. The closure also calculates the amounnt of 
    stock that the user needs to buy or sell short to hedge their position and plots 2 volatlity surfaces. 
    
    INPUTS: (user prompted to enter these in the following order)
    ======
    option type:   the user is prompted to enter option type or exit
    ticker:        accepts both upper and lower case (string), checks if its valid
    strike:        price at which the opion can be excercised (int)
    exp_year:      year of expiration (int), checks if its valid
    exp_month:     month of expiration (int), checks if its valid
    exp_day:       day of expiration (int), checks if its valid
    position:      number of long/short puts or calls for this stock that are on the books
    
    OUTPUT:
    ======
    delta Black Sholes:    deploying BS_Delta function above, Black Scholes delta is returned
    delta Bharadia:        deploying AutoDiff package partial derrivatives of implied vol are calculated
                           and delta estimate is returned
    delta Corrado:         deploying AutoDiff package partial derrivatives of implied vol are calculated
                           and delta estimate is returned
    hedging indstructions: whether the trader should buy or sell short and how much of the underlying asset.
    two plots:             volatlity surface plots for Bharadia and Corrado methods respectively
    
    NOTES:
    ====== 
    This package relies on yahoo_fin package for live stock prices and option price data as well as stock's 
    historical standard deviation calculated from the data pulled for the share price from the previous year. 
    If there are any issues with the yahoo_fin and yfinance pacakges being able to pull this data due to yahoo specific
    glitches, this extension package will not run.  In the 'real world' application, this package would be 
    linked to a more realiable platform like Bloomberg, which most of the traders use but is very expensive.
    Yahoo finance is free but slow annd not always reliable.
    """
    def ObtainInputs():
            #nested function to collect data from the user
            print("Please Select Type of Option to Evaluate")
            print("1) Exit")
            print("2) Puts")
            print("3) Calls")
            try:
                option=int(input(""))
            except: 
                print("Please enter number from the options above: ")
                option=int(input(""))
         
            #collect option type variable
            while option not in [1,2,3]:
                    option=int(input("Please enter number from the options above: "))
            if option==3:
                    op_type='calls'
            else:
                    if option==2:
                        op_type='puts'
                    else:
                        print('Thank you. Goodbye!')
                        return
 
            
            #collect ticker variable
            print("Please Enter Ticker")
            try:
                value = str(input(''))
                stock = yf.Ticker(value)#checking if the ticker is valid

                S=stock.history(period='min')['Open'][0]
                #si.get_live_price(value) #here we check if the ticker is valid
                
            except ValueError:
                print ("Sorry, {} is not a valid ticker, try again".format(value))
                ObtainInputs()
        
        
            ticker=value
            #collect strike value
            print("Please Enter Strike")
            try:
                option=int(input(""))
            except: ValueError ("Please enter integer value ")
            strike=option
            
            def Date():
                #nested function to organize data values
                today=date.today()
                print("Please Enter Expiration Year")
                try:
                    option=int(input(""))
                    if option<today.year or option > (today.year+2):
                        print("Please enter a valid listed year")
                        option=int(input(""))
                except: ValueError ("Please enter an integer value ")
                exp_year=option
            
                print("Please Enter Expiration Month")
                try:
                    value=int(input(""))
                except: ValueError ("Please enter valid date ")
                if value>12 or value<1:
                    print("Invalid input for Month. Please try again")
                    value=int(input(""))
                if exp_year==today.year and value<today.month:
                    print("Invalid input for Month. Please try again")
                    value=int(input(""))
                
                exp_month=value
            
                print("Please Enter Expiration Day")
                try:
                    option=int(input(""))
                    exp_day=option
                except: ValueError ("Please enter valid date ")
                if exp_month in [1,3,5,7,8,10,12]:
                    if option>31 or option <1:
                        print("Invalid input for day.  Please try again")
                        option=int(input(""))
                        
                else:
                    if exp_month==2 and exp_year==2020:
                        if option>29 or option <1:
                            print("Invalid input for day.  Please try again")
                            option=int(input(""))
                    else:
                        if exp_month in [4,6,9,11]:
                            if option>30 or option<1:
                                print("Invalid input for day.  Please try again")
                                option=int(input(""))  
               
               
                exp_dat=option
            
                exp_date=str(str(exp_month)+'/'+str(exp_day)+'/'+str(exp_year))
            
                return exp_year, exp_month, exp_day, exp_date
            #obtain the valid dates in the right format  
            exp_year, exp_month, exp_day, exp_date = Date()
            #calculate black scholes delta
            delta_bs, vega, T_t, S, C = BS_Delta(ticker, exp_year, exp_month, exp_day, op_type, strike) 
            
            #print output
            print("Black Scholes Delta: ", delta_bs)
            
            K = AutoDiff(strike, 1)  #####################--AUTODIFF USED HERE                         
            
            simple_implied = np.sqrt(2*np.pi/T_t) * ( ( C -(S - K)/2 ) / ( S - (S - K)/2 ) ) 
            deltaPi_simple_implied= simple_implied.derv
            complex_implied = np.sqrt(2*np.pi/T_t) * (1/(S + K)) *  ( C - ((S - K)/2)\
                                                                     + np.sqrt( (C - (S-K)/2)**2 - (S -K)**2/np.pi )) 
            deltaPi_complex_implied = complex_implied.derv                                            
            delta_simple=delta_bs+vega*deltaPi_simple_implied
            delta_complex=delta_bs+vega*deltaPi_complex_implied
            
            #print deltas after calculating them
            if delta_complex is None:
                delta_complex=delta_simple
                print('Could not approximate Corrado due to complex numbers ')
            print("Bharadia delta: ", delta_simple)
            print("Corrado delta: ", delta_complex)
            
            #check if the user would like to get delta hedging advice
            print("Would You Like to Delta Hedge Your Position: y/n?")
            try:
                option=str(input(""))
            except: raise ValueError("please entery 'y' or 'n'")
            if option=='y':
                print("How many units? Enter + values for long and - values for short")
                try:
                    option=int(input(""))
                except: raise ValueError('please enter an integer value')
                units=option
                if option >0 and op_type=='puts':
                        action = 'buy'
                elif option>0 and op_type=='calls':
                        action = 'sell short'
                elif option <0 and op_type=='calls':
                        action ='buy'
                elif option <0 and op_type=='puts':
                        action = 'sell short'
                recomend=abs(int(units*delta_bs))
                recomend1=abs(int(units*delta_simple))
                recomend2=abs(int(units*delta_complex))
                print("According to Black Scholes you need to ", action, "",abs(int(recomend)), " shares of ",ticker )
                print("According to Bharadia apporach you need to ", action, "",abs(int(recomend1)), " shares of ", ticker)
                print("Accoding to Corrado approach you need to ", action, "", abs(int(recomend2)), " shares of ", ticker)
                #plot 3D vol plots
                Volatility_Surface(ticker, exp_year, exp_month, exp_day, op_type, strike,C)    
                return
            else:
                #plot 3D vol polots
                Volatility_Surface(ticker, exp_year, exp_month, exp_day, op_type, strike, C)
                return
            
            return   
            
        
    ObtainInputs()
    
    
        
    
OptionsRun()
