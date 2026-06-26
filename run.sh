#nginx -t &&
#service nginx start &&
#service redis-server start && service redis-server status &&
cd project_contents && streamlit run digpt.py --theme.base "light"
