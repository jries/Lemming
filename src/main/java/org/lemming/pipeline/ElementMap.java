package org.lemming.pipeline;

import java.util.AbstractMap;
import java.util.Collections;
import java.util.Set;

import org.lemming.interfaces.Element;

public class ElementMap extends AbstractMap<String,Object> implements Element {
	
	private boolean isLast;
	private Set<Entry<String,Object>> entrySet;
	
	public ElementMap(Set<Entry<String,Object>> es){
		this.entrySet = es;
	}


	@Override
	public boolean isLast() {
		return isLast;
	}


	@Override
	public void setLast(boolean b) {
		isLast = b;		
	}

	@Override
	public Set<Entry<String,Object>> entrySet() {
        return Collections.unmodifiableSet(entrySet);
	}

	
}
