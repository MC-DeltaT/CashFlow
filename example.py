from calendar import MONDAY, SATURDAY, SUNDAY
from datetime import date

from cashflow.date_time import DateRange
from cashflow.frontend import Account, ExpenseSink, IncomeSource, ScheduleBuilder, tagset
from cashflow.probability import FloatDistribution
from cashflow.schedule import DayOfWeekDistribution, Monthly, Once, Weekdays, Weekly


# Schedule applies to the 2022/2023 financial year.
year_begin = date(2022, 7, 1)
year_end = date(2023, 6, 30)

all_year = DateRange.inclusive(year_begin, year_end)

# Declare bank accounts, investments, etc.
general_account = Account('Everyday account')       # Default account for income and expenses
savings_account = Account('Savings account', tagset('savings'))     # Tag this account as a "savings" account
investment_account = Account('Investment account', tagset('savings'))
accounts = (general_account, savings_account, investment_account)

schedule = ScheduleBuilder(general_account)

# Example: work paycheque of $5000 comes every month on the 15th day of month, occurring all year.
schedule.income('Work', Monthly(15), 5000)
# Example: tax return of $1000-$2000 comes once on 2022/8/1.
schedule.income('Tax return', Once(date(2022, 8, 1)), FloatDistribution.uniformly_in(1000, 2000))
# Example: public transport to work, occurs every weekday (Monday-Friday).
schedule.expense('Public transport', Weekdays(), 10)
# Example: rent is $150 every week, on Monday.
schedule.expense('Rent', Weekly(MONDAY), 300)
# Example: groceries cost $150 every week, on either Saturday or Sunday.
schedule.expense('Groceries', Weekly(DayOfWeekDistribution.uniformly_of(SATURDAY, SUNDAY)), 150)
# Example: save $1500 every month, on the 1st day of the month.
schedule.transfer('Savings', Monthly(1), 1500, general_account, savings_account)
# Example: invest $500 every 3 months.
schedule.transfer('Investment', Monthly(1, all_year, period=3), 500, general_account, investment_account)
# Other schedule types are available, see classes derived from `EventSchedule`.

# Timeframe in which we're doing the projection and analysis.
analysis_range = all_year

# Can leave these out to project change in account balances rather than absolute values.
initial_account_balances = {
    general_account: 5000,
    savings_account: 10000,
    investment_account: 15000
}


analysis = schedule.make_analysis(analysis_range)

# Print out a description of each cash flow event.
analysis.log_cash_flows()
print()

# Summarise total income, expenses, and savings.
analysis.summarise_cash_flows('income', lambda cash_flow: isinstance(cash_flow.source, IncomeSource))
analysis.summarise_cash_flows('expenses', lambda cash_flow: isinstance(cash_flow.sink, ExpenseSink))
analysis.summarise_cash_flows('savings', lambda cash_flow: isinstance(cash_flow.sink, Account) and 'savings' in cash_flow.sink.tags)

# Plot projected account balances over time.
analysis.plot_balances_over_time(accounts, initial_account_balances)
