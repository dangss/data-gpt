import pandas as _pd
import numpy as _np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

_OCCUPATION_LV0_NAME="occupation_color"
_OCCUPATION_LV1_NAME="major_occupation_group"
_OCCUPATION_LV2_NAME="minor_occupation_group"
_OCCUPATION_LV3_NAME="occupation_role"

import pathlib
root_dir = str(pathlib.Path(__file__).parent.resolve())
_AAID_MAP_PATH = root_dir+'/data/input_common/aaid_10_april.csv'


_LOCATION_PROVINCE_MAP = {
            0: 'Đồng Tháp',
            1: 'Bình Phước',
            2: 'Ninh Bình',
            3: 'Bạc Liêu',
            4: 'Hồ Chí Minh',
            5: 'Vĩnh Long',
            6: 'Lâm Đồng',
            7: 'Yên Bái',
            8: 'Hà Nam',
            9: 'Hà Nội',
            10: 'Hải Dương',
            11: 'Hậu Giang',
            12: 'An Giang',
            13: 'Trà Vinh',
            14: 'Tiền Giang',
            15: 'Tây Ninh',
            16: 'Đồng Nai',
            17: 'Đắk Lắk',
            18: 'Bình Định',
            19: 'Kon Tum',
            20: 'Đà Nẵng',
            21: 'Bắc Giang',
            22: 'Bắc Kạn',
            23: 'Điện Biên',
            24: 'Hòa Bình',
            25: 'Thái Bình',
            26: 'Vĩnh Phúc',
            27: 'Hà Giang',
            28: 'Kiên Giang',
            29: 'Bình Dương',
            30: 'Bình Thuận',
            31: 'Đắk Nông',
            32: 'Khánh Hòa',
            33: 'Gia Lai',
            34: 'Quảng Nam',
            35: 'Quảng Trị',
            36: 'Hà Tĩnh',
            37: 'Hưng Yên',
            38: 'Quảng Ninh',
            39: 'Thanh Hóa',
            40: 'Phú Thọ',
            41: 'Lai Châu',
            42: 'Thái Nguyên',
            43: 'Cao Bằng',
            44: 'Cà Mau',
            45: 'Cần Thơ',
            46: 'Sóc Trăng',
            47: 'Bến Tre',
            48: 'Long An',
            49: 'Bà Rịa - Vũng Tàu',
            50: 'Ninh Thuận',
            51: 'Phú Yên',
            52: 'Quảng Ngãi',
            53: 'Thừa Thiên Huế',
            54: 'Quảng Bình',
            55: 'Nghệ An',
            56: 'Nam Định',
            57: 'Hải Phòng',
            58: 'Lạng Sơn',
            59: 'Lào Cai',
            60: 'Sơn La',
            61: 'Bắc Ninh',
            62: 'Tuyên Quang'}

_OCCUPATION_COLOR_MAP = { 
            10: 'White',
            11: 'Religious',
            12: 'Legal',
            13: 'Education',
            14: 'Artist',
            15: 'Medical',
            16: 'Police',
            17: 'Service',
            18: 'Blue',
            19: 'Military',
            # 20: 'Worker',
            20: 'Blue',
            21: 'Housewife',
            22: 'Student'}

def _preprocess_occupation_name(df, lv0_col_name, lv1_col_name):
    df_tmp = df.copy()
    df_tmp[lv1_col_name] = (df_tmp[lv1_col_name].apply(
        lambda x: x.replace(" Occupations", "")
                .replace(" and Related", "")
                .replace(" Related", "")
                .replace(", and", ",")
                .replace(" and", ",") if x and type(x) == str else x))
    # hard code to handle duplicate major_occupation_group
    df_tmp.loc[(df_tmp[lv1_col_name] == 'Protective Service') & 
                      (df_tmp[lv0_col_name] == 'Blue'), lv1_col_name] = 'Protective Service '
    df_tmp.loc[(df_tmp[lv1_col_name] == 'Management Occupations') & 
                      (df_tmp[lv0_col_name] == 'Education'), lv1_col_name] = 'Management Occupations '
    return df_tmp


def parse_job_id_v2(df, col):
    df_occupation = df[[col]]
    df_occupation[_OCCUPATION_LV0_NAME] = df_occupation[col].apply(get_lv0_job_id).apply(get_color_name)
    df_occupation[_OCCUPATION_LV1_NAME] = df_occupation[col].apply(get_lv1_job_id).apply(get_job_name)
    df_occupation[_OCCUPATION_LV2_NAME] = df_occupation[col].apply(get_lv2_job_id).apply(get_job_name)
    df_occupation[_OCCUPATION_LV3_NAME] = df_occupation[col].apply(get_lv3_job_id).apply(get_job_name)
    df_occupation = _preprocess_occupation_name(df_occupation, _OCCUPATION_LV0_NAME, _OCCUPATION_LV1_NAME)
    df_occupation =  df_occupation.drop(columns=[col])
    return df_occupation


def parse_aaid(df, col_aaid):
    df_aaid = _pd.read_csv(_AAID_MAP_PATH)
    df_temp = df[[col_aaid]].merge(df_aaid, how='left', left_on=col_aaid, right_on='aaid')
    df_temp.loc[~df_temp.aaid.isnull() & df_temp.Country.isnull(), 'Country'] = 'Other Countries'
    return df_temp


def get_age_bin(age):
    if _np.isnan(age) or age is None:
        return _np.nan
    elif age < 18:
        return "<18"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age <= 65:
        return "55-65"
    elif age > 65:
        return ">65"
    else:
        return _np.nan


def get_phone_brand(device_name):
    if device_name == 'Null' or type(device_name) != str:
        return _np.nan
    elif 'ip' in device_name or 'apple' in device_name:
        return 'Apple'
    elif ' ' in device_name:
        return device_name.split(' ')[0].capitalize()
    elif 'rm-1040' in device_name or 'rm-1067' in device_name:
        return 'Microsoft'
    else:
        return 'Unkown'


def get_telco_from_noised_phone(noised_phone):
    telco_map = {
        'bgK2wz2': 'Viettel',
        'bgK2wj2': 'Viettel',
        'bgK2wj6': 'Viettel',
        'bgK2wjw': 'Viettel',
        'bgK2uDI': 'Viettel',
        'bgK2uDM': 'Viettel',
        'bgK2uDA': 'Viettel',
        'bgK2uDE': 'Viettel',
        'bgK2uD2': 'Viettel',
        'bgK2uD6': 'Viettel',
        'bgK2uDw': 'Viettel',
        'bgK2uD-': 'Viettel',
        'bgK2wzw': 'Vina',
        'bgK2wjU': 'Vina',
        'bgK2wjA': 'Vina',
        'bgK2wzM': 'Vina',
        'bgK2wzA': 'Vina',
        'bgK2wzE': 'Vina',
        'bgK2wzU': 'Vina',
        'bgK2wzI': 'Vina',
        'bgK2wjI': 'VietnamMobile',
        'bgK2vj2': 'VietnamMobile',
        'bgK2vjw': 'VietnamMobile',
        'bgK2wz-': 'MobiFone',
        'bgK2wjQ': 'MobiFone',
        'bgK2wjM': 'MobiFone',
        'bgK2vDQ': 'MobiFone',
        'bgK2vD-': 'MobiFone',
        'bgK2vD6': 'MobiFone',
        'bgK2vD2': 'MobiFone',
        'bgK2vDw': 'MobiFone',
        'bgK2wj-': 'gmobile',
        'bgK2vj-': 'gmobile',
        'bgK2vjI': 'VietnamMobile',
        'bgK2wz6': 'ITelecom',
        'bgK2vjE': 'Mobicast'
        }
    if noised_phone is None or noised_phone!=noised_phone:
        return _np.nan
    prefix = noised_phone[:7]
    return telco_map.get(prefix, 'Other_Countries')
        

def get_location_name(location):
    return _LOCATION_PROVINCE_MAP.get(location, _np.nan)


s1=u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0=u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
def remove_accents(input_str):
	s = ''
	for c in input_str:
		if c in s1:
			s += s0[s1.index(c)]
		else:
			s += c
	return s  


def get_location_tone_name(no_tone_names, map_path):
    # map no tone name into tone name
    map_province = _pd.read_csv(map_path, header=None)
    map_province.columns=['tone', 'no_tone']
    tone_names = []
    for no_tone_name in no_tone_names:
        tone_match = map_province[map_province.no_tone==no_tone_name]['tone']
        if len(tone_match) > 0:
            tone_names.append(tone_match.iloc[0])
        else:
            tone_names.append(no_tone_name)
    return tone_names


def get_lv0_job_id(job_id):
    return job_id//1e+6


def get_lv1_job_id(job_id):
    if job_id % 1e+6 == 0:
        return None
    else:
        return (job_id//1e+4*1e+4)%1e+6
    

def get_lv2_job_id(job_id):
    if job_id % 1e+4 == 0:
        return None
    else:
        return (job_id//1e+2*1e+2)%1e+6


def get_lv3_job_id(job_id):
    if job_id % 1e+2 == 0:
        return None
    else:
        return job_id


def get_color_name(color_code):
    return _OCCUPATION_COLOR_MAP.get(color_code, _np.nan)


def get_job_name(job_id):
    occupation_id_name_map = {
        110000: 'Management Occupations',
        111000: 'Top Executives',
        119030: 'Education and Childcare Administrators',
        130000: 'Business and Financial Operations Occupations',
        131020: 'Buyers and Purchasing Agents',
        131070: 'Human Resources Workers',
        131081: 'Logisticians',
        131160: 'Market Research Analysts and Marketing Specialists',
        132010: 'Accountants and Auditors',
        132020: 'Property Appraisers and Assessors',
        132054: 'Financial Risk Specialists',
        132072: 'Loan Officers',
        150000: 'Computer and Mathematical Occupations',
        152040: 'Statisticians',
        170000: 'Architecture and Engineering Occupations',
        171000: 'Architects, Surveyors, and Cartographers',
        172000: 'Engineers',
        172050: 'Civil Engineers',
        172070: 'Electrical and Electronics Engineers',
        210000: 'Community and Social Service Occupations',
        212000: 'Religious Workers',
        230000: 'Legal Occupations',
        231010: 'Lawyers and Judicial Law Clerks',
        231020: 'Judges, Magistrates, and Other Judicial Workers',
        250000: 'Educational Instruction and Library Occupations',
        251000: 'Postsecondary Teachers',
        252000: 'Preschool, Elementary, Middle, Secondary, and Special Education Teachers',
        252010: 'Preschool and Kindergarten Teachers',
        254000: 'Librarians, Curators, and Archivists',
        259090: 'Miscellaneous Educational Instruction and Library Workers',
        270000: 'Arts, Design, Entertainment, Sports, and Media Occupations',
        271000: 'Art and Design Workers',
        271010: 'Artists and Related Workers',
        271020: 'Designers',
        272000: 'Entertainers and Performers, Sports and Related Workers',
        272010: 'Actors, Producers, and Directors',
        272020: 'Athletes, Coaches, Umpires, and Related Workers',
        272030: 'Dancers and Choreographers',
        272040: 'Musicians, Singers, and Related Workers',
        273000: 'Media and Communication Workers',
        273040: 'Writers and Editors',
        274000: 'Media and Communication Equipment Workers',
        274020: 'Photographers',
        290000: 'Healthcare Practitioners and Technical Occupations',
        330000: 'Protective Service Occupations',
        331090: 'Miscellaneous First-Line Supervisors, Protective Service Workers',
        333050: 'Police Officers',
        350000: 'Food Preparation and Serving Related Occupations',
        351000: 'Supervisors of Food Preparation and Serving Workers',
        351010: 'Supervisors of Food Preparation and Serving Workers',
        352000: 'Cooks and Food Preparation Workers',
        353020: 'Fast Food and Counter Workers',
        370000: 'Building and Grounds Cleaning and Maintenance Occupations',
        390000: 'Personal Care and Service Occupations',
        395000: 'Personal Appearance Workers',
        395010: 'Barbers, Hairdressers, Hairstylists and Cosmetologists',
        395091: 'Makeup Artists, Theatrical and Performance',
        397000: 'Tour and Travel Guides',
        399030: 'Recreation and Fitness Workers',
        410000: 'Sales and Related Occupations',
        411111: 'Furniture',
        411112: 'Car',
        413020: 'Insurance Sales Agents',
        413030: 'Securities, Commodities, and Financial Services Sales Agents',
        419020: 'Real Estate Brokers and Sales Agents',
        419100: 'Wedding Clothes Sales',
        430000: 'Office and Administrative Support Occupations',
        434050: 'Customer Service Representatives',
        434080: 'Hotel, Motel, and Resort Desk Clerks',
        434180: 'Reservation and Transportation Ticket Agents and Travel Clerks',
        435050: 'Postal Service Workers',
        436000: 'Secretaries and Administrative Assistants',
        450000: 'Farming, Fishing, and Forestry Occupations',
        470000: 'Construction and Extraction Occupations',
        472110: 'Electricians',
        472060: 'Construction Laborers',
        490000: 'Installation, Maintenance, and Repair Occupations',
        510000: 'Production Occupations',
        515000: 'Printing Workers',
        517000: 'Woodworkers',
        530000: 'Transportation and Material Moving Occupations',
        530001: 'Driver Instructors',
        532000: 'Air Transportation Workers',
        537000: 'Material Moving Workers',
        550000: 'Military Specific Occupations',
        333000: 'Law Enforcement Workers',
        472000: 'Construction Trades Workers',
        131000: 'Business Operations Specialists',
        419000: 'Other Sales and Related Workers',
        411000: 'Supervisors of Sales Workers',
        132000: 'Financial Specialists',
        119000: 'Other Management Occupations',
        413000: 'Sales Representatives, Services',
        435000: 'Material Recording, Scheduling, Dispatching, and Distributing Workers',
        331000: 'Supervisors of Protective Service Workers',
        231000: 'Lawyers, Judges, and Related Workers',
        434000: 'Information and Record Clerks',
        399000: 'Other Personal Care and Service Workers',
        353000: 'Food and Beverage Serving Workers',
        259000: 'Other Educational Instruction and Library Occupations',
        152000: 'Mathematical Science Occupations'
    }
    return occupation_id_name_map.get(job_id, _np.nan)