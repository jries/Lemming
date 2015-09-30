package org.lemming.pipeline;

import java.util.HashMap;
import java.util.Map;

import org.lemming.interfaces.Element;

public class MapElement implements Element {
	
	private boolean isLast;
	private Map<String,Object> map = new HashMap<>();

	public MapElement(Map<String, Object> map) {
		this.map = map;
	}

	@Override
	public boolean isLast() {
		return isLast;
	}

	@Override
	public void setLast(boolean isLast) {
		this.isLast = isLast;
	}


	public Map<String,Object> get(){
		return map;
	}

}
