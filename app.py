from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# CORS Configuration - Allow requests from Loveable
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],  # In production, replace with your Loveable domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Store processed data in memory
processed_data = {
    'brand_presence': None,
    'sources_full': None,
    'sources_detailed': None,
    'sources_combined': None,
    'perception': None,
    'citation_tracker': None,
    'events_log': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_brand_presence(file_path):
    """Process the brand presence file with raw query data
    
    Brand Presence Calculation:
    - Each row in the input file represents one prompt response (query result)
    - is_present = True/False indicates if brand was mentioned in that response
    - Brand presence % = (# responses with brand mention) / (total # responses) * 100
    - This is calculated using .mean() on boolean is_present field:
      * .mean() on booleans gives the proportion of True values (0.0 to 1.0)
      * Multiply by 100 to get percentage
    - Calculated separately for:
      * Overall (all LLMs, all topics, by week)
      * Per LLM (by week)
      * Per Topic (by week)
      * Per LLM + Topic combination (by week)
    
    Example: If on 2024-01-01 there were 10 queries for ChatGPT on "AI Safety":
      - 3 mentioned the brand (is_present=True)
      - 7 did not (is_present=False)
      - Brand presence = 3/10 * 100 = 30.0%
    """
    import json
    from collections import Counter, defaultdict
    
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from brand presence file")
        print(f"Columns: {df.columns.tolist()}")
        
        # Extract brand name by correlating rank to ordered_brands
        brand_name = None
        if 'rank' in df.columns and 'ordered_brands' in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row['rank']) and row['rank'] > 0 and pd.notna(row['ordered_brands']):
                    try:
                        brands = json.loads(row['ordered_brands'])
                        if len(brands) >= int(row['rank']):
                            brand_name = brands[int(row['rank']) - 1]
                            print(f"Extracted brand name: {brand_name}")
                            break
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        
        # Filter out Claude from analysis
        df = df[df['llm_name'].str.lower() != 'claude'].copy()
        print(f"After filtering out Claude: {len(df)} rows")
        
        # Normalize LLM names
        def normalize_llm_name(name):
            if pd.isna(name):
                return name
            name_lower = str(name).lower().strip()
            
            llm_mapping = {
                'aio': 'Google AIO',
                'ai overview': 'Google AIO',
                'google aio': 'Google AIO',
                'chatgpt': 'ChatGPT',
                'chat gpt': 'ChatGPT',
                'gpt': 'ChatGPT',
                'gemini': 'Gemini',
                'perplexity': 'Perplexity',
                'claude': 'Claude'
            }
            
            return llm_mapping.get(name_lower, name)
        
        df['llm_name'] = df['llm_name'].apply(normalize_llm_name)
        print(f"Normalized LLM names: {df['llm_name'].unique().tolist()}")
        
        # Convert analysis_date to datetime
        df['analysis_date'] = pd.to_datetime(df['analysis_date'])
        df['week_start'] = df['analysis_date']
        df['is_present'] = df['is_present'].astype(bool)
        
        # Handle topic column naming - rename topic_name to topic if it exists
        if 'topic_name' in df.columns and 'topic' not in df.columns:
            df['topic'] = df['topic_name']
            print("Renamed 'topic_name' column to 'topic'")
        elif 'topic' not in df.columns:
            print("Warning: No 'topic' or 'topic_name' column found. Adding default 'General' topic.")
            df['topic'] = 'General'
        
        # Overall daily trend
        weekly_trend = df.groupby('analysis_date').agg({
            'is_present': 'mean'
        }).reset_index()
        weekly_trend['brand_presence'] = weekly_trend['is_present'] * 100
        weekly_trend = weekly_trend.sort_values('analysis_date')
        weekly_trend = weekly_trend.rename(columns={'analysis_date': 'week_start'})
        
        # LLM performance over time
        llm_weekly_trend = df.groupby(['analysis_date', 'llm_name']).agg({
            'is_present': 'mean'
        }).reset_index()
        llm_weekly_trend['brand_presence'] = llm_weekly_trend['is_present'] * 100
        llm_weekly_trend = llm_weekly_trend.sort_values('analysis_date')
        llm_weekly_trend = llm_weekly_trend.rename(columns={'analysis_date': 'week_start'})
        
        # Average rank by LLM over time
        llm_rank_trend = df[df['is_present'] == True].groupby(['analysis_date', 'llm_name']).agg({
            'rank': 'mean'
        }).reset_index()
        llm_rank_trend = llm_rank_trend.sort_values('analysis_date')
        llm_rank_trend = llm_rank_trend.rename(columns={'analysis_date': 'week_start'})
        
        # Get last week and previous week data
        if len(weekly_trend) >= 2:
            last_week = weekly_trend.iloc[-1]
            prev_week = weekly_trend.iloc[-2]
            current_presence = last_week['brand_presence']
            prev_presence = prev_week['brand_presence']
            week_change = current_presence - prev_presence
            latest_date = last_week['week_start']
        else:
            current_presence = weekly_trend.iloc[-1]['brand_presence'] if len(weekly_trend) > 0 else 0
            week_change = 0
            latest_date = weekly_trend.iloc[-1]['week_start'] if len(weekly_trend) > 0 else None
        
        # LLM breakdown for the latest week
        if latest_date is not None:
            latest_llm_data = df[df['analysis_date'] == latest_date].groupby('llm_name').agg({
                'is_present': 'mean'
            }).reset_index()
            latest_llm_data['brand_presence'] = latest_llm_data['is_present'] * 100
            latest_llm_data = latest_llm_data.sort_values('brand_presence', ascending=False)
        else:
            latest_llm_data = pd.DataFrame()
        
        # Topic breakdown for the latest week
        if latest_date is not None:
            latest_topic_data = df[df['analysis_date'] == latest_date].groupby('topic').agg({
                'is_present': 'mean'
            }).reset_index()
            latest_topic_data['brand_presence'] = latest_topic_data['is_present'] * 100
            latest_topic_data = latest_topic_data.sort_values('brand_presence', ascending=False)
        else:
            latest_topic_data = pd.DataFrame()
        
        # Topic performance over time
        topic_weekly_trend = df.groupby(['analysis_date', 'topic']).agg({
            'is_present': 'mean'
        }).reset_index()
        topic_weekly_trend['brand_presence'] = topic_weekly_trend['is_present'] * 100
        topic_weekly_trend = topic_weekly_trend.sort_values('analysis_date')
        topic_weekly_trend = topic_weekly_trend.rename(columns={'analysis_date': 'week_start'})
        
        # Topic + LLM combination analysis
        topic_by_llm_data = {}
        for llm in df['llm_name'].unique():
            if pd.notna(llm):
                llm_df = df[df['llm_name'] == llm]
                topic_by_llm_data[llm] = {}
                
                for topic in llm_df['topic'].unique():
                    if pd.notna(topic):
                        topic_df = llm_df[llm_df['topic'] == topic]
                        topic_trend = topic_df.groupby('analysis_date').agg({
                            'is_present': 'mean'
                        }).reset_index()
                        topic_trend['brand_presence'] = topic_trend['is_present'] * 100
                        topic_trend = topic_trend.sort_values('analysis_date')
                        
                        topic_by_llm_data[llm][topic] = {
                            'dates': topic_trend['analysis_date'].dt.strftime('%Y-%m-%d').tolist(),
                            'presence': topic_trend['brand_presence'].tolist()
                        }
        
        # Topic + LLM rank analysis
        topic_rank_by_llm_data = {}
        for llm in df['llm_name'].unique():
            if pd.notna(llm):
                llm_df = df[df['llm_name'] == llm]
                topic_rank_by_llm_data[llm] = {}
                
                for topic in llm_df['topic'].unique():
                    if pd.notna(topic):
                        topic_df = llm_df[llm_df['topic'] == topic]
                        topic_df = topic_df[topic_df['is_present'] == True]
                        
                        topic_rank_trend = topic_df.groupby('analysis_date').agg({
                            'rank': 'mean'
                        }).reset_index()
                        topic_rank_trend = topic_rank_trend.sort_values('analysis_date')
                        
                        if len(topic_rank_trend) > 0:
                            topic_rank_by_llm_data[llm][topic] = {
                                'dates': topic_rank_trend['analysis_date'].dt.strftime('%Y-%m-%d').tolist(),
                                'rank': topic_rank_trend['rank'].tolist()
                            }
        
        # Get all competitors from ordered_brands
        all_competitors = set()
        for _, row in df.iterrows():
            if pd.notna(row.get('ordered_brands')):
                try:
                    brands = json.loads(row['ordered_brands'])
                    all_competitors.update(brands)
                except (json.JSONDecodeError, TypeError):
                    continue
        
        # Remove the main brand if found
        if brand_name:
            all_competitors.discard(brand_name)
        
        all_competitors = sorted(list(all_competitors))
        
        # Competitor analysis data structure
        competitor_data = {}
        for competitor in all_competitors:
            competitor_trends = {}
            
            for date in df['analysis_date'].unique():
                date_df = df[df['analysis_date'] == date]
                
                competitor_count = 0
                total_responses = 0
                
                for _, row in date_df.iterrows():
                    if pd.notna(row.get('ordered_brands')):
                        try:
                            brands = json.loads(row['ordered_brands'])
                            total_responses += 1
                            if competitor in brands:
                                competitor_count += 1
                        except (json.JSONDecodeError, TypeError):
                            continue
                
                if total_responses > 0:
                    presence_pct = (competitor_count / total_responses) * 100
                    competitor_trends[pd.Timestamp(date).strftime('%Y-%m-%d')] = presence_pct
            
            competitor_data[competitor] = competitor_trends
        
        # Prepare data for JSON serialization
        return {
            'brand_name': brand_name,
            'latest_date': latest_date.strftime('%Y-%m-%d') if latest_date else None,
            'current_presence': float(current_presence),
            'week_change': float(week_change),
            'weekly_trend': {
                'dates': weekly_trend['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                'presence': weekly_trend['brand_presence'].tolist()
            },
            'llm_weekly_trend': {
                llm: {
                    'dates': llm_weekly_trend[llm_weekly_trend['llm_name'] == llm]['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                    'presence': llm_weekly_trend[llm_weekly_trend['llm_name'] == llm]['brand_presence'].tolist()
                }
                for llm in llm_weekly_trend['llm_name'].unique()
            },
            'llm_rank_trend': {
                llm: {
                    'dates': llm_rank_trend[llm_rank_trend['llm_name'] == llm]['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                    'rank': llm_rank_trend[llm_rank_trend['llm_name'] == llm]['rank'].tolist()
                }
                for llm in llm_rank_trend['llm_name'].unique()
            },
            'latest_llm_breakdown': {
                'llms': latest_llm_data['llm_name'].tolist() if not latest_llm_data.empty else [],
                'presence': latest_llm_data['brand_presence'].tolist() if not latest_llm_data.empty else []
            },
            'latest_topic_breakdown': {
                'topics': latest_topic_data['topic'].tolist() if not latest_topic_data.empty else [],
                'presence': latest_topic_data['brand_presence'].tolist() if not latest_topic_data.empty else []
            },
            'topic_weekly_trend': {
                topic: {
                    'dates': topic_weekly_trend[topic_weekly_trend['topic'] == topic]['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                    'presence': topic_weekly_trend[topic_weekly_trend['topic'] == topic]['brand_presence'].tolist()
                }
                for topic in topic_weekly_trend['topic'].unique()
            },
            'topic_by_llm_data': topic_by_llm_data,
            'topic_rank_by_llm_data': topic_rank_by_llm_data,
            'available_dates': sorted(df['analysis_date'].dt.strftime('%Y-%m-%d').unique().tolist()),
            'all_competitors': all_competitors,
            'competitor_data': competitor_data
        }
        
    except Exception as e:
        print(f"Error processing brand presence file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_sources_full(file_path):
    """Process sources full export"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from sources full file")
        
        # Convert date column
        df['week_start'] = pd.to_datetime(df['week_start'])
        
        # Basic stats
        total_queries = len(df)
        queries_with_sources = df['has_sources'].sum() if 'has_sources' in df.columns else 0
        
        # Weekly trend
        weekly_stats = df.groupby('week_start').agg({
            'has_sources': 'mean' if 'has_sources' in df.columns else lambda x: 0
        }).reset_index()
        weekly_stats['source_rate'] = weekly_stats.get('has_sources', 0) * 100
        
        return {
            'total_queries': int(total_queries),
            'queries_with_sources': int(queries_with_sources),
            'weekly_trend': {
                'dates': weekly_stats['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                'source_rate': weekly_stats['source_rate'].tolist()
            }
        }
    except Exception as e:
        print(f"Error processing sources full file: {str(e)}")
        raise

def process_sources_detailed(file_path):
    """Process sources detailed export"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from sources detailed file")
        
        # Aggregate by domain
        domain_stats = df.groupby('domain').agg({
            'appearances': 'sum'
        }).reset_index().sort_values('appearances', ascending=False).head(20)
        
        return {
            'top_domains': {
                'domains': domain_stats['domain'].tolist(),
                'appearances': domain_stats['appearances'].tolist()
            }
        }
    except Exception as e:
        print(f"Error processing sources detailed file: {str(e)}")
        raise

def process_perception(file_path):
    """Process perception analysis file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from perception file")
        
        # Convert date
        df['week_start'] = pd.to_datetime(df['week_start'])
        
        # Sentiment analysis
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        
        # Weekly sentiment trend
        weekly_sentiment = df.groupby(['week_start', 'sentiment']).size().reset_index(name='count')
        weekly_sentiment_pivot = weekly_sentiment.pivot(
            index='week_start', 
            columns='sentiment', 
            values='count'
        ).fillna(0)
        
        return {
            'sentiment_counts': sentiment_counts,
            'weekly_sentiment': {
                'dates': weekly_sentiment_pivot.index.strftime('%Y-%m-%d').tolist(),
                'sentiments': {
                    col: weekly_sentiment_pivot[col].tolist()
                    for col in weekly_sentiment_pivot.columns
                }
            }
        }
    except Exception as e:
        print(f"Error processing perception file: {str(e)}")
        raise

def process_citation_tracker(file_path):
    """Process citation tracker file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from citation tracker file")
        
        # Convert date
        df['week_start'] = pd.to_datetime(df['week_start'])
        
        # Citation stats
        total_citations = len(df)
        unique_sources = df['domain'].nunique() if 'domain' in df.columns else 0
        
        # Weekly citation trend
        weekly_citations = df.groupby('week_start').size().reset_index(name='count')
        
        return {
            'total_citations': int(total_citations),
            'unique_sources': int(unique_sources),
            'weekly_citations': {
                'dates': weekly_citations['week_start'].dt.strftime('%Y-%m-%d').tolist(),
                'count': weekly_citations['count'].tolist()
            }
        }
    except Exception as e:
        print(f"Error processing citation tracker file: {str(e)}")
        raise

def process_events_log(file_path):
    """Process events log file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows from events log file")
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        return {
            'events': df.to_dict('records')
        }
    except Exception as e:
        print(f"Error processing events log file: {str(e)}")
        raise

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'AEO Dashboard API is running'})

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Upload and process data files"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        file_type = request.form.get('file_type')
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process file based on type
        if file_type == 'brand_presence':
            processed_data['brand_presence'] = process_brand_presence(filepath)
        elif file_type == 'sources_full':
            processed_data['sources_full'] = process_sources_full(filepath)
        elif file_type == 'sources_detailed':
            processed_data['sources_detailed'] = process_sources_detailed(filepath)
        elif file_type == 'perception':
            processed_data['perception'] = process_perception(filepath)
        elif file_type == 'citation_tracker':
            processed_data['citation_tracker'] = process_citation_tracker(filepath)
        elif file_type == 'events_log':
            processed_data['events_log'] = process_events_log(filepath)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Clean up file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': f'{file_type} file processed successfully',
            'file_type': file_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/brand_presence', methods=['GET'])
def get_brand_presence():
    """Get brand presence data"""
    if processed_data['brand_presence'] is None:
        return jsonify({'error': 'No brand presence data available'}), 404
    
    return jsonify(processed_data['brand_presence'])

@app.route('/api/data/sources_full', methods=['GET'])
def get_sources_full():
    """Get sources full data"""
    if processed_data['sources_full'] is None:
        return jsonify({'error': 'No sources full data available'}), 404
    
    return jsonify(processed_data['sources_full'])

@app.route('/api/data/sources_detailed', methods=['GET'])
def get_sources_detailed():
    """Get sources detailed data"""
    if processed_data['sources_detailed'] is None:
        return jsonify({'error': 'No sources detailed data available'}), 404
    
    return jsonify(processed_data['sources_detailed'])

@app.route('/api/data/perception', methods=['GET'])
def get_perception():
    """Get perception data"""
    if processed_data['perception'] is None:
        return jsonify({'error': 'No perception data available'}), 404
    
    return jsonify(processed_data['perception'])

@app.route('/api/data/citation_tracker', methods=['GET'])
def get_citation_tracker():
    """Get citation tracker data"""
    if processed_data['citation_tracker'] is None:
        return jsonify({'error': 'No citation tracker data available'}), 404
    
    return jsonify(processed_data['citation_tracker'])

@app.route('/api/data/events_log', methods=['GET'])
def get_events_log():
    """Get events log data"""
    if processed_data['events_log'] is None:
        return jsonify({'error': 'No events log data available'}), 404
    
    return jsonify(processed_data['events_log'])

@app.route('/api/data/all', methods=['GET'])
def get_all_data():
    """Get all available data"""
    return jsonify({
        'brand_presence': processed_data['brand_presence'],
        'sources_full': processed_data['sources_full'],
        'sources_detailed': processed_data['sources_detailed'],
        'perception': processed_data['perception'],
        'citation_tracker': processed_data['citation_tracker'],
        'events_log': processed_data['events_log']
    })

@app.route('/api/data/status', methods=['GET'])
def get_data_status():
    """Get status of all data sources"""
    return jsonify({
        'brand_presence': processed_data['brand_presence'] is not None,
        'sources_full': processed_data['sources_full'] is not None,
        'sources_detailed': processed_data['sources_detailed'] is not None,
        'perception': processed_data['perception'] is not None,
        'citation_tracker': processed_data['citation_tracker'] is not None,
        'events_log': processed_data['events_log'] is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
