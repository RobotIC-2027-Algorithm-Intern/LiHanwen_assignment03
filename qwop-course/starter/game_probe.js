// 只读取原游戏的物理状态，不修改重力、碰撞或身体结构。
window.__rlSnapshot = function () {
  const g = window.__qwop.game;
  const contacts = new Set();
  for (let c = g.m_world.m_contactList; c; c = c.getNext()) {
    if (!c.isTouching()) continue;
    const a = c.getFixtureA().getBody().getUserData();
    const b = c.getFixtureB().getBody().getUserData();
    if (a === 'track') contacts.add(b);
    if (b === 'track') contacts.add(a);
  }
  const forbidden = ['leftCalf', 'rightCalf', 'leftThigh', 'rightThigh',
                     'torso', 'head', 'leftArm', 'rightArm', 'leftForearm', 'rightForearm'];
  return [globalbodystate, {...globalgamestate,
    left_knee_y: g.leftKnee.m_bodyA.getWorldPoint(g.leftKnee.m_localAnchor1).y,
    right_knee_y: g.rightKnee.m_bodyA.getWorldPoint(g.rightKnee.m_localAnchor1).y,
    left_foot_contact: Number(contacts.has('leftFoot')),
    right_foot_contact: Number(contacts.has('rightFoot')),
    nonfoot_contact: Number(forbidden.some(p => contacts.has(p))),
    ground_contacts: [...contacts]
  }];
};
