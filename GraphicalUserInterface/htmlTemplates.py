css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  width: 78px;
  height: 78px;
  border-radius: 50%;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.button-disabled {
    opacity: 0.5;
    cursor: not-allowed;
}
.stForm {
    position: relative;
}
.stForm .stButton {
    position: absolute;
    right: 0;
}
.chat-container {
    max-height: 70vh;
    overflow-y: auto;
}
</style>
<script>
document.addEventListener("DOMContentLoaded", function() {
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
</script>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="/avatars/RobotBook.jpg" style="width: 78px; height: 78px; border-radius: 50%;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="/avatars/Person.jpg" style="width: 78px; height: 78px; border-radius: 50%;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
