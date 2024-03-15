from flask import Flask, render_template, request, flash,get_flashed_messages,Response,session,redirect,url_for, send_from_directory, send_file,abort
import base64
from PIL import Image
import torch
from torchvision import transforms
from model import Encoder, Generator
import io
import os
from key import secret_key,salt,salt2
from itsdangerous import URLSafeTimedSerializer
from stoken import token
from cmail import sendmail
import mysql.connector.pooling

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model weights
netE = Encoder().to(device)
netG = Generator().to(device)
netE.load_state_dict(torch.load("netE.model", map_location=device))
netG.load_state_dict(torch.load("netG.model", map_location=device))
netE.eval()
netG.eval()

app = Flask(__name__)
app.secret_key = secret_key
app.config['SESSION_TYPE']='filesystem'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)),'static')

# db=os.environ['RDS_DB_NAME']
# user=os.environ['RDS_USERNAME']
# password=os.environ['RDS_PASSWORD']
# host=os.environ['RDS_HOSTNAME']
# port=os.environ['RDS_PORT']

# conn=mysql.connector.pooling.MySQLConnectionPool(host=host,user=user,password=password,db=db,port=port,pool_name='DED',pool_size=3,pool_reset_session=True)

conn=mysql.connector.pooling.MySQLConnectionPool(host='localhost',user='root',password="admin",db='image',pool_name='DED',pool_size=3, pool_reset_session=True)

try:
    mydb=conn.get_connection()
    cursor = mydb.cursor(buffered=True)
    cursor.execute('CREATE TABLE IF NOT EXISTS users (uid INT PRIMARY KEY auto_increment, username VARCHAR(50), password VARCHAR(15), email VARCHAR(60))')

except Exception as e:
    print(e)
finally:
    if mydb.is_connected():
        mydb.close()


@app.route('/login',methods=['GET','POST'])
def login():
    if session.get('username'):
        return redirect(url_for('home'))
    if request.method=='POST':
        # print(request.form)
        name=request.form['name']
        password=request.form['password']
        try:
            mydb=conn.get_connection()
            cursor=mydb.cursor(buffered=True)
        except Exception as e:
            print(e)
        else:
            cursor.execute('SELECT count(*) from users where username=%s and password=%s',[name,password])
            count=cursor.fetchone()[0]
            cursor.close()
            if count==1:
                session['username']=name
                return redirect(url_for('home'))
            else:
                flash('Invalid username or password')
                return render_template('login.html')
        finally:
            if mydb.is_connected():
                mydb.close()
    return render_template('login.html')

@app.route('/registration',methods=['GET','POST'])
def registration():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        email=request.form['email']
        try:
            mydb=conn.get_connection()
            cursor=mydb.cursor(buffered=True)
        except Exception as e:
            print(e)
        else:
            cursor.execute('SELECT COUNT(*) FROM users WHERE username = %s', [username])
            count=cursor.fetchone()[0]
            cursor.execute('select count(*) from users where email=%s',[email])
            count1=cursor.fetchone()[0]
            cursor.close()
            if count==1:
                flash('username already in use')
                return render_template('registration.html')
            elif count1==1:
                flash('Email already in use')
                return render_template('registration.html')
            data={'username':username,'password':password,'email':email}
            subject='Email Confirmation'
            body=f"Thanks for signing up\n\nfollow this link for further steps-{url_for('confirm',token=token(data,salt),_external=True)}"
            sendmail(to=email,subject=subject,body=body)
            flash('Confirmation link sent to mail')
            return redirect(url_for('login'))
        finally:
            if mydb.is_connected():
                mydb.close()
    return render_template('registration.html')

@app.route('/confirm/<token>')
def confirm(token):
    try:
        serializer=URLSafeTimedSerializer(secret_key)
        data=serializer.loads(token,salt=salt,max_age=180)
    except Exception as e:
        #print(e)
        return 'Link Expired register again'
    else:
        try:
            mydb=conn.get_connection()
            cursor=mydb.cursor(buffered=True)
        except Exception as e:
            print(e)
        else:
            username=data['username']
            cursor.execute('select count(*) from users where username=%s',[username])
            count=cursor.fetchone()[0]
            if count==1:
                cursor.close()
                flash('You are already registerterd!')
                return redirect(url_for('login'))
            else:
                cursor.execute('insert into users(username,password,email) values(%s,%s,%s)',(data['username'], data['password'], data['email']))
                mydb.commit()
                cursor.close()
                flash('Details registered!')
                return redirect(url_for('login'))
        finally:
            if mydb.is_connected():
                mydb.close()


@app.route('/forget',methods=['GET','POST'])
def forgot():
    if request.method=='POST':
        email=request.form['email']
        try:
            mydb=conn.get_connection()
            cursor=mydb.cursor(buffered=True)
        except Exception as e:
            print(e)
        else:
            cursor.execute('select count(*) from users where email=%s',[email])
            count=cursor.fetchone()[0]
            cursor.close()
            if count==1:
                cursor=mydb.cursor(buffered=True)
                cursor.execute('SELECT email from users where email=%s',[email])
                status=cursor.fetchone()[0]
                cursor.close()
                subject='Forget Password'
                confirm_link=url_for('reset',token=token(email,salt=salt2),_external=True)
                body=f"Use this link to reset your password-\n\n{confirm_link}"
                sendmail(to=email,body=body,subject=subject)
                flash('Reset link sent check your email')
                return redirect(url_for('login'))
            else:
                flash('Invalid email id')
                return render_template('forgot.html')
        finally:
            if mydb.is_connected():
                mydb.close()
    return render_template('forgot.html')


@app.route('/reset/<token>',methods=['GET','POST'])
def reset(token):
    try:
        serializer=URLSafeTimedSerializer(secret_key)
        email=serializer.loads(token,salt=salt2,max_age=180)
    except:
        abort(404,'Link Expired')
    else:
        if request.method=='POST':
            newpassword=request.form['npassword']
            confirmpassword=request.form['cpassword']
            if newpassword==confirmpassword:
                try:
                    mydb=conn.get_connection()
                    cursor=mydb.cursor(buffered=True)
                except Exception as e:
                    print(e)
                else:
                    cursor.execute('update users set password=%s where email=%s',[newpassword,email])
                    mydb.commit()
                    flash('Reset Successful')
                    return redirect(url_for('login'))
                finally:
                    if mydb.is_connected():
                        mydb.close()
            else:
                flash('Passwords mismatched')
                return render_template('newpassword.html')
        return render_template('newpassword.html')

@app.route('/logout')
def logout():
    if session.get('username'):
        session.pop('username')
        flash('Successfully logged out')
        return redirect(url_for('login'))



@app.route('/')
def home():
    if session.get('username'):
        return render_template('home.html')
    else:
        return redirect(url_for('login'))

@app.route('/compressImage', methods=['GET','POST'])
def index():
    if session.get('username'):
        if 'image' not in request.files:
            return 'No image uploaded', 400

        uploaded_file = request.files['image']

        if uploaded_file.filename == '':
            return 'No image selected', 400

        # Save the original image
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg')
        uploaded_file.save(original_image_path)

        original_image = Image.open(uploaded_file).convert('RGB')
        original_size = original_image.size

        # Resize the original image to match the model dimensions
        target_size = (218, 178)  # Use the dimensions from your model
        image = original_image.resize(target_size)

        # Preprocess the image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Compress and reconstruct the image
        with torch.no_grad():
            encoded_img = netE(image_tensor)
            reconstructed_img = netG(encoded_img).cpu()

        # Postprocess the reconstructed image
        reconstructed_img = (reconstructed_img.squeeze() * 0.5 + 0.5).clamp(0, 1)
        reconstructed_img = transforms.ToPILImage()(reconstructed_img)

        # Resize the compressed image to match the original dimensions
        compressed_image = reconstructed_img.resize(original_size)

        # Save the compressed image to the server
        compressed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_image.jpg')
        compressed_image.save(compressed_image_path)

        # Convert the compressed image to base64 for displaying on the web page
        with open(compressed_image_path, "rb") as image_file:
            compressed_image_data = base64.b64encode(image_file.read()).decode('utf-8')

        return render_template('index.html', compressed_image_data=compressed_image_data)
    else:
        return redirect(url_for('login'))

@app.route('/static/<path:path>')
def serve_static(path):
    if session.get('username'):
        return send_from_directory('static', path)
    else:
        return redirect(url_for('login'))

@app.route('/download_compressed_image')
def download_compressed_image():
    if session.get('username'):
        compressed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_image.jpg')
        if not os.path.exists(compressed_image_path):
            return 'No compressed image available', 404
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'compressed_image.jpg', as_attachment=True)
    else:
        return redirect(url_for('login'))
if __name__ == '__main__':
    app.run(debug=True)