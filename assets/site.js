
document.querySelectorAll('.mobile-toggle').forEach(button=>{
  button.addEventListener('click',()=>{
    const nav=document.querySelector('.nav-links');
    const open=nav.classList.toggle('open');
    button.setAttribute('aria-expanded',String(open));
  });
});

document.querySelectorAll('[data-tabs]').forEach(shell=>{
  const buttons=[...shell.querySelectorAll('[role="tab"]')];
  const panels=[...shell.querySelectorAll('[role="tabpanel"]')];
  function activate(button){
    buttons.forEach(b=>b.setAttribute('aria-selected',String(b===button)));
    panels.forEach(p=>p.hidden=p.id!==button.getAttribute('aria-controls'));
    button.focus({preventScroll:true});
  }
  buttons.forEach((button,index)=>{
    button.addEventListener('click',()=>activate(button));
    button.addEventListener('keydown',event=>{
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      let next=index;
      if(event.key==='ArrowRight') next=(index+1)%buttons.length;
      if(event.key==='ArrowLeft') next=(index-1+buttons.length)%buttons.length;
      if(event.key==='Home') next=0;
      if(event.key==='End') next=buttons.length-1;
      activate(buttons[next]);
    });
  });
});
