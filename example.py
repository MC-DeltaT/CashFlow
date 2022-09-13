from calendar import MONDAY, SATURDAY, SUNDAY
from datetime import date

from cashflow.analysis import generate_cash_flow_logs, plot_funds_over_time, simulate_cash_flows, summarise_cash_flows
from cashflow.core import Account
from cashflow.datetime import DateRange
from cashflow.helpers import ExpenseSink, IncomeSource, ScheduleBuilder, tagset
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

# Can leave these out to get change in account balances rather than absolute values.
initial_account_balances = {
    general_account: 5000,
    savings_account: 10000,
    investment_account: 15000
}


# Generate all possible cash flows occurring in the analysis timeframe.
cash_flows = list(schedule.iterate_occurrences(analysis_range))

# Simulate the cash flows, which gives us a projection of account balances over time.
account_balances = simulate_cash_flows(cash_flows, accounts, analysis_range, initial_account_balances)

# Print out a description of each cash flow event.
for log in generate_cash_flow_logs(cash_flows):
    print(log)
print()

# Summarise total income, expenses, and savings.
print(summarise_cash_flows(
    [occurrence for occurrence in cash_flows if isinstance(occurrence.event.cash_flow.source, IncomeSource)], 'income'))
print(summarise_cash_flows(
    [occurrence for occurrence in cash_flows if isinstance(occurrence.event.cash_flow.sink, ExpenseSink)], 'expenses'))
print(summarise_cash_flows(
    [occurrence for occurrence in cash_flows if 'savings' in occurrence.event.cash_flow.sink.tags], 'savings'))

# Plot account balances over time, plus our net savings.
plot_funds_over_time(account_balances, ('Net savings', lambda account: 'savings' in account.tags))
