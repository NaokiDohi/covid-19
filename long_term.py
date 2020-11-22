#long term predictions
x_long_term = np.arange('2020-10-22', '2021-03-30', dtype='datetime64[D]').astype('datetime64[D]')#11-21
long_term_future = len(x_long_term)
print(str(long_term_future) + '日間後')

sns.set()
COVID = plt.figure(figsize=(20,8))
plt.title("COVID-19 in Japan after " + str(long_term_future) + ' days', y=-0.15)
plt.grid(True)
plt.xlabel("Date")
plt.ylabel("Nunber of Person infected with corona virus (people)")
plt.plot(x_all,data_at_japan_diff,'g',lw=3,label='daily_at_japan')
plt.plot(x_long_term, predictions_infected_pepole_long_term, 'r',lw=3,alpha=0.7,label='upcoming_future')
plt.legend(loc='upper left')
plt.show()
