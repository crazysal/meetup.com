use meetup;
select m.member_id, m.bio,  m.group_id, 
g.group_name, g.description as group_description, `category.name` as category_name, g.join_mode, g.members as members_count, g.rating as group_rating, g.visibility, `organizer.member_id` as organizer_id,
e.event_id, e.event_name, e.description as event_description, e.duration, `fee.accepts` as fee_accepts, `fee.amount` as fee_amount, `fee.required`, e.yes_rsvp_count,
v.venue_id, v.venue_name, v.normalised_rating as venue_rating, v.rating_count as venue_rating_count   
from members as m
inner join 
groups as g
on m.group_id = g.group_id
inner join
events as e
on g.group_id = e.group_id 
inner join venues as v
on v.venue_id = e.venue_id
; 