import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
import io
import os

# Set the style for matplotlib
plt.style.use('fivethirtyeight')

# Create directory for images if it doesn't exist
os.makedirs('/home/ubuntu/trading_system/report/images', exist_ok=True)

# 1. Momentum Strategy Visualization
def create_momentum_visualization():
    # Sample data for momentum strategy
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    market = 100 + np.cumsum(np.random.normal(0.05, 1, 100))
    momentum = 100 + np.cumsum(np.random.normal(0.08, 1.2, 100))
    value = 100 + np.cumsum(np.random.normal(0.03, 0.9, 100))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(dates, market, label='Market Index', linewidth=2)
    plt.plot(dates, momentum, label='Momentum Strategy', linewidth=2)
    plt.plot(dates, value, label='Value Strategy', linewidth=2)
    
    # Add a vertical line to indicate market regime change
    plt.axvline(x=dates[50], color='red', linestyle='--', alpha=0.7, label='Market Regime Change')
    
    # Highlight momentum outperformance period
    plt.axvspan(dates[20], dates[40], alpha=0.2, color='green', label='Strong Momentum Period')
    
    # Add labels and title
    plt.title('Momentum Strategy Performance Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Performance (Base 100)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format the date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add annotations
    plt.annotate('Momentum Outperformance', xy=(dates[30], momentum[30]), 
                 xytext=(dates[30], momentum[30]+10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/momentum_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Sector Rotation Heatmap
def create_sector_rotation_heatmap():
    # Sample data for sector performance
    sectors = ['Technology', 'Healthcare', 'Financials', 'Energy', 'Consumer', 
               'Industrials', 'Materials', 'Utilities', 'Real Estate', 'Communication']
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Generate random performance data with some patterns
    np.random.seed(42)  # For reproducibility
    data = np.random.normal(0, 3, (len(sectors), len(months)))
    
    # Add some patterns to make it more realistic
    data[0, :3] += 5  # Tech outperforms in Q1
    data[3, 3:6] += 4  # Energy outperforms in Q2
    data[1, 6:9] += 3  # Healthcare outperforms in Q3
    data[2, 9:] += 4   # Financials outperforms in Q4
    
    # Create a DataFrame
    df = pd.DataFrame(data, index=sectors, columns=months)
    
    # Create a custom colormap (green for positive, red for negative)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', 
              '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Create the heatmap
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(df, cmap=cmap, annot=True, fmt=".1f", linewidths=.5, 
                     cbar_kws={'label': 'Monthly Return (%)'})
    
    # Add title and labels
    plt.title('Sector Performance Heatmap (Monthly Returns %)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Sector', fontsize=12)
    
    # Highlight the best performing sector each month
    for j in range(len(months)):
        i = np.argmax(df.iloc[:, j].values)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
    # Add annotations for rotation strategy
    plt.figtext(0.5, 0.01, 'Sector Rotation Strategy: Invest in the top 2 performing sectors based on 3-month momentum', 
                ha='center', fontsize=12, bbox={'facecolor':'lightgrey', 'alpha':0.5, 'pad':5})
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/sector_rotation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Swing Trading Pattern Visualization
def create_swing_trading_visualization():
    # Sample data for a stock price
    trading_days = 120
    dates = pd.date_range(start='2024-01-01', periods=trading_days, freq='B')
    
    # Create a price pattern with some swings
    np.random.seed(42)
    price = 100
    prices = [price]
    
    for i in range(1, trading_days):
        if i < 20:
            change = np.random.normal(0.05, 1.0)
        elif i < 40:
            change = np.random.normal(-0.1, 1.2)
        elif i < 60:
            change = np.random.normal(0.2, 1.0)
        elif i < 80:
            change = np.random.normal(-0.05, 1.1)
        else:
            change = np.random.normal(0.15, 1.0)
        
        price = price * (1 + change/100)
        prices.append(price)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot the price
    plt.plot(dates, prices, label='Stock Price', linewidth=2)
    
    # Highlight swing trade opportunities
    plt.axvspan(dates[15], dates[25], alpha=0.2, color='red', label='Swing Trade 1 (Short)')
    plt.axvspan(dates[55], dates[65], alpha=0.2, color='green', label='Swing Trade 2 (Long)')
    plt.axvspan(dates[95], dates[105], alpha=0.2, color='green', label='Swing Trade 3 (Long)')
    
    # Add entry and exit points
    entry_points = [dates[15], dates[55], dates[95]]
    exit_points = [dates[25], dates[65], dates[105]]
    entry_prices = [prices[15], prices[55], prices[95]]
    exit_prices = [prices[25], prices[65], prices[105]]
    
    plt.scatter(entry_points, entry_prices, color='blue', s=100, marker='^', label='Entry Point')
    plt.scatter(exit_points, exit_prices, color='purple', s=100, marker='v', label='Exit Point')
    
    # Add annotations
    plt.annotate('Entry: Breakout', xy=(entry_points[0], entry_prices[0]), 
                 xytext=(entry_points[0], entry_prices[0]*1.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10)
    
    plt.annotate('Exit: Target Reached', xy=(exit_points[0], exit_prices[0]), 
                 xytext=(exit_points[0], exit_prices[0]*0.95),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10)
    
    # Add technical indicators
    sma_20 = pd.Series(prices).rolling(window=20).mean()
    sma_50 = pd.Series(prices).rolling(window=50).mean()
    
    plt.plot(dates, sma_20, label='20-day SMA', linestyle='--', linewidth=1.5)
    plt.plot(dates, sma_50, label='50-day SMA', linestyle='--', linewidth=1.5)
    
    # Add labels and title
    plt.title('Swing Trading Pattern Identification', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format the date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=10))
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/swing_trading_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Risk Management Decision Tree
def create_risk_management_tree():
    # Create a figure
    plt.figure(figsize=(16, 10))
    
    # Define the coordinates for the nodes
    nodes = {
        'root': (8, 9),
        'market_up': (4, 7),
        'market_down': (12, 7),
        'sector_strong': (2, 5),
        'sector_weak': (6, 5),
        'sector_strong_down': (10, 5),
        'sector_weak_down': (14, 5),
        'full_pos': (1, 3),
        'reduced_pos': (3, 3),
        'small_pos': (5, 3),
        'no_pos': (7, 3),
        'small_short': (9, 3),
        'medium_short': (11, 3),
        'large_short': (13, 3),
        'full_short': (15, 3)
    }
    
    # Define the connections between nodes
    connections = [
        ('root', 'market_up'),
        ('root', 'market_down'),
        ('market_up', 'sector_strong'),
        ('market_up', 'sector_weak'),
        ('market_down', 'sector_strong_down'),
        ('market_down', 'sector_weak_down'),
        ('sector_strong', 'full_pos'),
        ('sector_strong', 'reduced_pos'),
        ('sector_weak', 'small_pos'),
        ('sector_weak', 'no_pos'),
        ('sector_strong_down', 'small_short'),
        ('sector_strong_down', 'medium_short'),
        ('sector_weak_down', 'large_short'),
        ('sector_weak_down', 'full_short')
    ]
    
    # Draw the connections
    for start, end in connections:
        plt.plot([nodes[start][0], nodes[end][0]], [nodes[start][1], nodes[end][1]], 'k-', linewidth=1.5)
    
    # Draw the nodes
    node_colors = {
        'root': 'lightblue',
        'market_up': 'lightgreen',
        'market_down': 'lightcoral',
        'sector_strong': 'lightgreen',
        'sector_weak': 'khaki',
        'sector_strong_down': 'khaki',
        'sector_weak_down': 'lightcoral',
        'full_pos': 'green',
        'reduced_pos': 'lightgreen',
        'small_pos': 'palegreen',
        'no_pos': 'white',
        'small_short': 'pink',
        'medium_short': 'lightcoral',
        'large_short': 'indianred',
        'full_short': 'red'
    }
    
    for node, (x, y) in nodes.items():
        plt.scatter(x, y, s=1000, color=node_colors[node], edgecolors='black', zorder=10)
    
    # Add labels to the nodes
    node_labels = {
        'root': 'Market Analysis',
        'market_up': 'Market Uptrend\n(Above 200-day MA)',
        'market_down': 'Market Downtrend\n(Below 200-day MA)',
        'sector_strong': 'Sector Strong\n(Top 3 Momentum)',
        'sector_weak': 'Sector Weak\n(Bottom 7 Momentum)',
        'sector_strong_down': 'Sector Resilient\n(Relative Strength)',
        'sector_weak_down': 'Sector Weak\n(Falling Rapidly)',
        'full_pos': 'Full Position\n(100% Allocation)',
        'reduced_pos': 'Reduced Position\n(75% Allocation)',
        'small_pos': 'Small Position\n(25% Allocation)',
        'no_pos': 'No Position\n(Cash)',
        'small_short': 'Small Short\n(25% Allocation)',
        'medium_short': 'Medium Short\n(50% Allocation)',
        'large_short': 'Large Short\n(75% Allocation)',
        'full_short': 'Full Short\n(100% Allocation)'
    }
    
    for node, (x, y) in nodes.items():
        plt.annotate(node_labels[node], (x, y), ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add decision criteria
    decision_labels = {
        ('root', 'market_up'): 'Market > 200-day MA',
        ('root', 'market_down'): 'Market < 200-day MA',
        ('market_up', 'sector_strong'): 'Top 3 Momentum',
        ('market_up', 'sector_weak'): 'Bottom 7 Momentum',
        ('market_down', 'sector_strong_down'): 'RS > 1.0',
        ('market_down', 'sector_weak_down'): 'RS < 1.0',
        ('sector_strong', 'full_pos'): 'Vol < 20',
        ('sector_strong', 'reduced_pos'): 'Vol > 20',
        ('sector_weak', 'small_pos'): 'Improving',
        ('sector_weak', 'no_pos'): 'Deteriorating',
        ('sector_strong_down', 'small_short'): 'Improving',
        ('sector_strong_down', 'medium_short'): 'Deteriorating',
        ('sector_weak_down', 'large_short'): 'Vol < 30',
        ('sector_weak_down', 'full_short'): 'Vol > 30'
    }
    
    for start, end in connections:
        mid_x = (nodes[start][0] + nodes[end][0]) / 2
        mid_y = (nodes[start][1] + nodes[end][1]) / 2
        
        # Add a small offset to avoid overlapping with the line
        offset_x = 0.3 if nodes[start][0] < nodes[end][0] else -0.3
        
        plt.annotate(decision_labels.get((start, end), ''), 
                     (mid_x + offset_x, mid_y), 
                     ha='center', va='center', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     fontsize=8)
    
    # Add title and remove axes
    plt.title('Risk Management Decision Tree', fontsize=16)
    plt.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/risk_management_tree.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. Crypto Volatility Comparison
def create_crypto_volatility_comparison():
    # Sample data for volatility comparison
    assets = ['Bitcoin', 'Ethereum', 'S&P 500', 'Gold', 'NASDAQ', 'Apple', 'Tesla']
    
    # Annualized volatility (%)
    volatility = [65, 85, 15, 12, 20, 35, 60]
    
    # Create a color map based on volatility
    colors = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fdae61', '#f46d43', '#d73027']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(assets, volatility, color=colors)
    
    # Add a horizontal line for average stock volatility
    plt.axhline(y=20, color='black', linestyle='--', alpha=0.7, label='Average Stock Volatility')
    
    # Add labels and title
    plt.title('Asset Volatility Comparison (Annualized)', fontsize=16)
    plt.xlabel('Asset', fontsize=12)
    plt.ylabel('Annualized Volatility (%)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height}%', ha='center', va='bottom', fontsize=10)
    
    # Add risk categories
    plt.annotate('Low Volatility', xy=(3, 5), xytext=(3, 5), ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#e0f3f8", ec="gray", alpha=0.8),
                 fontsize=10)
    
    plt.annotate('Medium Volatility', xy=(4, 25), xytext=(4, 25), ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fdae61", ec="gray", alpha=0.8),
                 fontsize=10)
    
    plt.annotate('High Volatility', xy=(1, 75), xytext=(1, 75), ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#d73027", ec="gray", alpha=0.8),
                 fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/crypto_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. System Architecture Diagram
def create_system_architecture():
    # Create a figure
    plt.figure(figsize=(16, 12))
    
    # Define the components
    components = {
        'kb': (5, 8, 3, 2, 'Trading Knowledge Base'),
        'agent': (5, 5, 3, 2, 'AI Trading Agent'),
        'algo': (5, 2, 3, 2, 'Algorithmic Trading System'),
        'data': (1, 5, 3, 2, 'Data Analysis & Monitoring'),
        'ib_api': (9, 2, 3, 2, 'Interactive Brokers API'),
        'market': (13, 2, 3, 2, 'Financial Markets'),
        'kaizen': (9, 8, 3, 2, 'Continuous Improvement')
    }
    
    # Define the connections
    connections = [
        ('kb', 'agent', 'Knowledge Retrieval'),
        ('agent', 'algo', 'Trading Decisions'),
        ('algo', 'ib_api', 'Order Execution'),
        ('ib_api', 'market', 'Market Access'),
        ('data', 'agent', 'Market Analysis'),
        ('data', 'kb', 'Data Storage'),
        ('algo', 'kb', 'Trade Records'),
        ('kaizen', 'kb', 'Strategy Updates'),
        ('kaizen', 'agent', 'Model Refinement'),
        ('ib_api', 'data', 'Market Data')
    ]
    
    # Draw the components
    component_colors = {
        'kb': '#c6dbef',
        'agent': '#9ecae1',
        'algo': '#6baed6',
        'data': '#4292c6',
        'ib_api': '#2171b5',
        'market': '#084594',
        'kaizen': '#fdae61'
    }
    
    for comp, (x, y, width, height, label) in components.items():
        rect = plt.Rectangle((x, y), width, height, facecolor=component_colors[comp], 
                            edgecolor='black', linewidth=2, alpha=0.8)
        plt.gca().add_patch(rect)
        plt.text(x + width/2, y + height/2, label, ha='center', va='center', 
                fontsize=12, fontweight='bold')
    
    # Draw the connections
    connection_styles = {
        'Knowledge Retrieval': 'arc3,rad=0.1',
        'Trading Decisions': 'arc3,rad=0',
        'Order Execution': 'arc3,rad=0',
        'Market Access': 'arc3,rad=0',
        'Market Analysis': 'arc3,rad=0.2',
        'Data Storage': 'arc3,rad=0.2',
        'Trade Records': 'arc3,rad=0.3',
        'Strategy Updates': 'arc3,rad=-0.2',
        'Model Refinement': 'arc3,rad=-0.3',
        'Market Data': 'arc3,rad=-0.4'
    }
    
    for start, end, label in connections:
        start_x = components[start][0] + components[start][2]/2
        start_y = components[start][1] + components[start][3]/2
        end_x = components[end][0] + components[end][2]/2
        end_y = components[end][1] + components[end][3]/2
        
        plt.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', connectionstyle=connection_styles[label],
                                    color='black', linewidth=1.5))
        
        # Calculate the midpoint of the connection
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Add a small offset based on the connection style
        if 'rad' in connection_styles[label]:
            rad = float(connection_styles[label].split('=')[1])
            offset_x = -rad * 3
            offset_y = rad * 3
        else:
            offset_x = 0
            offset_y = 0
        
        plt.text(mid_x + offset_x, mid_y + offset_y, label, ha='center', va='center',
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add title and remove axes
    plt.title('DeepResearch Trading System Architecture', fontsize=18)
    plt.axis('off')
    
    # Add legend for component types
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#c6dbef', edgecolor='black', label='Data Storage'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#9ecae1', edgecolor='black', label='Decision Making'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#6baed6', edgecolor='black', label='Execution'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#4292c6', edgecolor='black', label='Analysis'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#2171b5', edgecolor='black', label='External Interface'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#084594', edgecolor='black', label='External Systems'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#fdae61', edgecolor='black', label='Improvement Process')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              ncol=4, fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

# 7. Implementation Timeline
def create_implementation_timeline():
    # Create a figure
    plt.figure(figsize=(16, 8))
    
    # Define the tasks and their durations
    tasks = [
        'Infrastructure Setup',
        'Knowledge Base Development',
        'AI Agent Development',
        'Algorithmic Trading System',
        'Data Analysis & Monitoring',
        'Continuous Improvement',
        'Integration & Testing',
        'Paper Trading',
        'Live Deployment',
        'Performance Monitoring'
    ]
    
    # Define the start and duration for each task (in weeks)
    start_times = [0, 2, 4, 6, 4, 8, 10, 12, 14, 14]
    durations = [3, 4, 6, 6, 6, 4, 4, 4, 2, 10]
    
    # Define the colors for each task
    colors = plt.cm.viridis(np.linspace(0, 1, len(tasks)))
    
    # Create the Gantt chart
    for i, (task, start, duration, color) in enumerate(zip(tasks, start_times, durations, colors)):
        plt.barh(i, duration, left=start, height=0.5, color=color, alpha=0.8, 
                edgecolor='black', linewidth=1)
        
        # Add task label
        plt.text(start + duration/2, i, task, ha='center', va='center', 
                fontsize=10, fontweight='bold', color='black')
    
    # Add milestones
    milestones = [
        (3, 'Infrastructure Ready'),
        (8, 'Core Components Ready'),
        (12, 'System Integration Complete'),
        (16, 'Live Trading Begins')
    ]
    
    for week, label in milestones:
        plt.axvline(x=week, color='red', linestyle='--', alpha=0.7)
        plt.text(week, len(tasks) + 0.5, label, ha='center', va='center', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    # Add labels and title
    plt.title('Implementation Timeline (Weeks)', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.yticks([])  # Hide y-axis labels since we have them in the bars
    plt.grid(True, alpha=0.3, axis='x')
    
    # Set x-axis limits
    plt.xlim(0, 24)
    
    # Add week numbers
    for week in range(0, 25, 2):
        plt.text(week, -0.5, str(week), ha='center', va='center', fontsize=9)
    
    # Add phases
    phases = [
        (0, 4, 'Phase 1: Setup'),
        (4, 10, 'Phase 2: Development'),
        (10, 14, 'Phase 3: Testing'),
        (14, 24, 'Phase 4: Deployment & Monitoring')
    ]
    
    for start, end, label in phases:
        plt.axvspan(start, end, ymin=0.95, ymax=1.0, alpha=0.3, color='gray')
        plt.text((start + end)/2, len(tasks) + 1.2, label, ha='center', va='center', 
                fontsize=11, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/implementation_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Position Sizing Visualization
def create_position_sizing_visualization():
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Define the variables
    account_sizes = np.linspace(100000, 1000000, 100)
    risk_percentages = [0.5, 1.0, 2.0, 3.0]
    stop_loss_percentages = [5, 10, 15]
    
    # Create the plot
    for stop_loss in stop_loss_percentages:
        for risk in risk_percentages:
            position_sizes = (account_sizes * risk/100) / (stop_loss/100)
            plt.plot(account_sizes, position_sizes, 
                    label=f'Risk: {risk}%, Stop-Loss: {stop_loss}%',
                    linewidth=2)
    
    # Add labels and title
    plt.title('Position Sizing Based on Account Size and Risk Parameters', fontsize=16)
    plt.xlabel('Account Size ($)', fontsize=12)
    plt.ylabel('Position Size ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format the axes
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'${int(x/1000)}K'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f'${int(y/1000)}K'))
    
    # Add annotations
    plt.annotate('Conservative', xy=(300000, 15000), xytext=(300000, 50000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, ha='center')
    
    plt.annotate('Aggressive', xy=(300000, 120000), xytext=(300000, 180000),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, ha='center')
    
    # Add the formula
    formula = r'Position Size = $\frac{\text{Account Size} \times \text{Risk Percentage}}{\text{Stop-Loss Percentage}}$'
    plt.text(0.5, 0.05, formula, ha='center', va='center', fontsize=14,
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/position_sizing.png', dpi=300, bbox_inches='tight')
    plt.close()

# 9. KPI Dashboard
def create_kpi_dashboard():
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sample data for KPIs
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 1. Monthly Returns
    returns = [3.2, 2.1, -1.5, 4.3, 5.2, -0.8, 2.9, 6.1, -2.3, 3.7, 4.5, 5.8]
    benchmark = [1.5, 1.2, -2.0, 2.1, 2.5, -1.5, 1.8, 2.2, -3.0, 1.9, 2.3, 2.7]
    
    ax1 = axs[0, 0]
    ax1.bar(months, returns, color='#4292c6', alpha=0.7, label='Strategy')
    ax1.plot(months, benchmark, marker='o', color='#d73027', linewidth=2, label='Benchmark')
    
    ax1.set_title('Monthly Returns (%)', fontsize=14)
    ax1.set_xlabel('Month', fontsize=10)
    ax1.set_ylabel('Return (%)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add value labels
    for i, v in enumerate(returns):
        ax1.text(i, v + 0.3 if v > 0 else v - 0.7, f'{v}%', ha='center', fontsize=9)
    
    # 2. Cumulative Returns
    cumulative_returns = np.cumprod(np.array(returns)/100 + 1) - 1
    cumulative_benchmark = np.cumprod(np.array(benchmark)/100 + 1) - 1
    
    ax2 = axs[0, 1]
    ax2.plot(months, cumulative_returns * 100, marker='o', color='#4292c6', linewidth=2, label='Strategy')
    ax2.plot(months, cumulative_benchmark * 100, marker='o', color='#d73027', linewidth=2, label='Benchmark')
    
    ax2.set_title('Cumulative Returns (%)', fontsize=14)
    ax2.set_xlabel('Month', fontsize=10)
    ax2.set_ylabel('Cumulative Return (%)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # 3. Risk Metrics
    metrics = ['Sharpe\nRatio', 'Sortino\nRatio', 'Max\nDrawdown', 'Win\nRate', 'Profit\nFactor']
    values = [1.8, 2.2, -12.5, 68, 2.5]
    benchmark_values = [0.7, 0.9, -18.0, 55, 1.3]
    
    ax3 = axs[1, 0]
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, values, width, color='#4292c6', alpha=0.7, label='Strategy')
    ax3.bar(x + width/2, benchmark_values, width, color='#d73027', alpha=0.7, label='Benchmark')
    
    ax3.set_title('Risk Metrics', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    # Add value labels
    for i, v in enumerate(values):
        ax3.text(i - width/2, v + 0.1 if v > 0 else v - 0.7, f'{v}', ha='center', fontsize=9)
    
    for i, v in enumerate(benchmark_values):
        ax3.text(i + width/2, v + 0.1 if v > 0 else v - 0.7, f'{v}', ha='center', fontsize=9)
    
    # 4. Asset Allocation
    assets = ['Tech', 'Healthcare', 'Financials', 'Energy', 'Consumer', 'Cash']
    allocation = [30, 20, 15, 10, 15, 10]
    
    ax4 = axs[1, 1]
    wedges, texts, autotexts = ax4.pie(allocation, labels=assets, autopct='%1.1f%%',
                                      startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(assets))))
    
    ax4.set_title('Current Asset Allocation', fontsize=14)
    
    # Make the labels more readable
    for text in texts:
        text.set_fontsize(10)
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_weight('bold')
    
    # Add a summary of key metrics
    plt.figtext(0.5, 0.01, 'Key Performance Summary: Total Return: 32.5% | Alpha: 18.2% | Beta: 0.85 | Sharpe: 1.8', 
                ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", fc="#e0f3f8", ec="gray", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Add a title for the entire dashboard
    plt.suptitle('Trading Strategy Performance Dashboard', fontsize=18, y=0.98)
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/kpi_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

# 10. AI Decision Making Process
def create_ai_decision_process():
    # Create a figure
    plt.figure(figsize=(16, 10))
    
    # Define the components and their positions
    components = [
        (1, 8, 'Market Data\nCollection', '#c6dbef'),
        (4, 8, 'Feature\nEngineering', '#9ecae1'),
        (7, 8, 'Pattern\nRecognition', '#6baed6'),
        (10, 8, 'Prediction\nModels', '#4292c6'),
        (13, 8, 'Decision\nRules', '#2171b5'),
        (16, 8, 'Trade\nExecution', '#084594'),
        
        (2.5, 6, 'Technical\nIndicators', '#c6dbef'),
        (5.5, 6, 'Sentiment\nAnalysis', '#9ecae1'),
        (8.5, 6, 'Momentum\nScoring', '#6baed6'),
        (11.5, 6, 'Risk\nAssessment', '#4292c6'),
        (14.5, 6, 'Position\nSizing', '#2171b5'),
        
        (4, 4, 'Historical\nPerformance', '#c6dbef'),
        (8, 4, 'Current Market\nConditions', '#6baed6'),
        (12, 4, 'Strategy\nParameters', '#2171b5'),
        
        (8, 2, 'Feedback Loop &\nContinuous Learning', '#fdae61')
    ]
    
    # Define the connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # Main flow
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10),  # Downward connections
        (6, 11), (7, 11), (8, 12), (9, 12), (10, 13),  # Lower level connections
        (11, 14), (12, 14), (13, 14),  # Feedback connections
        (14, 2), (14, 3), (14, 4)  # Feedback to upper levels
    ]
    
    # Draw the components
    for i, (x, y, label, color) in enumerate(components):
        circle = plt.Circle((x, y), 0.8, facecolor=color, edgecolor='black', alpha=0.8)
        plt.gca().add_patch(circle)
        plt.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw the connections
    for start, end in connections:
        start_x, start_y = components[start][0], components[start][1]
        end_x, end_y = components[end][0], components[end][1]
        
        # Calculate the angle for the arrow
        angle = np.arctan2(end_y - start_y, end_x - start_x)
        
        # Adjust start and end points to be on the circle edge
        start_x += 0.8 * np.cos(angle)
        start_y += 0.8 * np.sin(angle)
        end_x -= 0.8 * np.cos(angle)
        end_y -= 0.8 * np.sin(angle)
        
        plt.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                    color='black', linewidth=1.5))
    
    # Add data flow labels
    data_flows = [
        (2.5, 8.5, 'Price & Volume Data'),
        (5.5, 8.5, 'Technical Indicators'),
        (8.5, 8.5, 'Pattern Signals'),
        (11.5, 8.5, 'Probability Scores'),
        (14.5, 8.5, 'Trade Signals')
    ]
    
    for x, y, label in data_flows:
        plt.text(x, y, label, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add title and remove axes
    plt.title('AI Trading Agent Decision Process', fontsize=18)
    plt.axis('off')
    
    # Add legend for process stages
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#c6dbef', markersize=15, label='Data Collection'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9ecae1', markersize=15, label='Processing'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6baed6', markersize=15, label='Analysis'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4292c6', markersize=15, label='Prediction'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2171b5', markersize=15, label='Decision'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#084594', markersize=15, label='Execution'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#fdae61', markersize=15, label='Learning')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              ncol=7, fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('/home/ubuntu/trading_system/report/images/ai_decision_process.png', dpi=300, bbox_inches='tight')
    plt.close()

# Generate all visualizations
create_momentum_visualization()
create_sector_rotation_heatmap()
create_swing_trading_visualization()
create_risk_management_tree()
create_crypto_volatility_comparison()
create_system_architecture()
create_implementation_timeline()
create_position_sizing_visualization()
create_kpi_dashboard()
create_ai_decision_process()

print("All visualizations have been created successfully!")
